using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Transform
{
    /// <summary>
    /// Provides highly optimized single-precision FFT operations for Complex32 numbers.
    /// </summary>
    public static class VortexFFT
    {
        // =========================================================================================
        // Pre-computed constants
        // =========================================================================================
        private const int MaxTableDepth = 20;
        private static readonly double[][] _realRootsOfUnity;
        private static readonly double[][] _imaginaryRootsOfUnity;
        private static readonly double[][] _imaginaryInverseRootsOfUnity;
        private static readonly int _simdWidth =
            Vector512.IsHardwareAccelerated ? 8 :
            Vector256.IsHardwareAccelerated ? 4 :
            Vector128.IsHardwareAccelerated ? 2 :
            0;

        static VortexFFT()
        {
            _realRootsOfUnity = new double[MaxTableDepth][];
            _imaginaryRootsOfUnity = new double[MaxTableDepth][];
            _imaginaryInverseRootsOfUnity = new double[MaxTableDepth][];

            for (int i = 0; i < MaxTableDepth; i++)
            {
                int n = 1 << i;
                _realRootsOfUnity[i] = new double[n];
                _imaginaryRootsOfUnity[i] = new double[n];
                _imaginaryInverseRootsOfUnity[i] = new double[n];

                double angleStep = -2.0 * Math.PI / n;
                for (int k = 0; k < n; k++)
                {
                    double angle = k * angleStep;
                    (double sin, double cos) = Math.SinCos(angle);
                    _realRootsOfUnity[i][k] = cos;
                    _imaginaryRootsOfUnity[i][k] = sin;
                    _imaginaryInverseRootsOfUnity[i][k] = -sin;
                }
            }
        }

        // =========================================================================================
        // Public API
        // =========================================================================================

        /// <summary>
        /// Computes the Fast Fourier Transform (FFT) of a complex array.
        /// </summary>
        public static void Fft(Complex[] data) => Fft(data.AsSpan());

        /// <summary>
        /// Computes the Fast Fourier Transform (FFT) of a complex span.
        /// </summary>
        public static void Fft(Span<Complex> data)
        {
            int n = data.Length;
            if (n <= 1) return;
            if (!BitOperations.IsPow2(n))
                throw new ArgumentException("Data length must be a power of two.");

            double[]? poolArray = null;
            Span<double> buffer = n <= 4096
                ? stackalloc double[n * 2]
                : (poolArray = ArrayPool<double>.Shared.Rent(n * 2));

            try
            {
                Span<double> realParts = buffer[..n];
                Span<double> imaginaryParts = buffer.Slice(n, n);

                // 1. Combined operation: Deinterleave + Bit Reverse
                DeinterleaveAndBitReverse(data, realParts, imaginaryParts);

                // 2. Core computation
                FftCore(realParts, imaginaryParts, isForward: true);

                // 3. Interleave and write back
                InterleaveAndWriteBack(realParts, imaginaryParts, data);
            }
            finally
            {
                if (poolArray != null)
                    ArrayPool<double>.Shared.Return(poolArray);
            }
        }

        /// <summary>
        /// Computes the Inverse Fast Fourier Transform (IFFT) of a complex span.
        /// </summary>
        public static void InverseFft(Span<Complex> data)
        {
            int n = data.Length;
            if (n == 0) return;

            double[]? poolArray = null;
            Span<double> buffer = n <= 4096
                ? stackalloc double[n * 2]
                : (poolArray = ArrayPool<double>.Shared.Rent(n * 2));

            try
            {
                Span<double> realParts = buffer[..n];
                Span<double> imaginaryParts = buffer.Slice(n, n);

                DeinterleaveAndBitReverse(data, realParts, imaginaryParts);

                FftCore(realParts, imaginaryParts, isForward: false);

                double scale = 1.0 / n;
                InterleaveAndWriteBackWithScale(realParts, imaginaryParts, data, scale);
            }
            finally
            {
                if (poolArray != null)
                    ArrayPool<double>.Shared.Return(poolArray);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void FftCore(Span<double> realParts, Span<double> imaginaryParts, bool isForward)
        {
            int n = realParts.Length;
            int logN = BitOperations.Log2((uint)n);

            var realRootsTable = _realRootsOfUnity;
            var imaginaryRootsTable = isForward ? _imaginaryRootsOfUnity : _imaginaryInverseRootsOfUnity;

            ref double realBase = ref MemoryMarshal.GetReference(realParts);
            ref double imaginaryBase = ref MemoryMarshal.GetReference(imaginaryParts);

            if (logN >= 1)
                ProcessStageScalar(ref realBase, ref imaginaryBase, n, 1,
                    realRootsTable[1], imaginaryRootsTable[1]);

            if (logN >= 2)
                ProcessStageScalar(ref realBase, ref imaginaryBase, n, 2,
                    realRootsTable[2], imaginaryRootsTable[2]);

            for (int stage = 3; stage <= logN; stage++)
            {
                int m2 = 1 << (stage - 1);

                if (_simdWidth == 8 && m2 >= 8)
                {
                    ProcessStageVector512(ref realBase, ref imaginaryBase, n, stage,
                        realRootsTable[stage], imaginaryRootsTable[stage]);
                }
                else if (_simdWidth == 4 && m2 >= 4)
                {
                    ProcessStageVector256(ref realBase, ref imaginaryBase, n, stage,
                        realRootsTable[stage], imaginaryRootsTable[stage]);
                }
                else if (_simdWidth == 2 && m2 >= 2)
                {
                    ProcessStageVector128(ref realBase, ref imaginaryBase, n, stage,
                        realRootsTable[stage], imaginaryRootsTable[stage]);
                }
                else
                {
                    ProcessStageScalar(ref realBase, ref imaginaryBase, n, stage,
                        realRootsTable[stage], imaginaryRootsTable[stage]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ProcessStageVector512(
            ref double realBase,
            ref double imaginaryBase,
            int n,
            int stage,
            double[] realRoots,
            double[] imaginaryRoots)
        {
            int blockSize = 1 << stage;
            int halfBlockSize = blockSize >> 1;
            const int VectorSize = 8;

            ref Vector512<double> realRootVectorBase = ref Unsafe.As<double, Vector512<double>>(
                ref MemoryMarshal.GetArrayDataReference(realRoots));
            ref Vector512<double> imaginaryRootVectorBase = ref Unsafe.As<double, Vector512<double>>(
                ref MemoryMarshal.GetArrayDataReference(imaginaryRoots));

            int vectorizedOperations = halfBlockSize / VectorSize;

            for (int blockStart = 0; blockStart < n; blockStart += blockSize)
            {
                for (int v = 0; v < vectorizedOperations; v++)
                {
                    int evenIndex = blockStart + v * VectorSize;
                    int oddIndex = evenIndex + halfBlockSize;

                    Vector512<double> evenReal = Unsafe.ReadUnaligned<Vector512<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, evenIndex)));
                    Vector512<double> evenImaginary = Unsafe.ReadUnaligned<Vector512<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)));
                    Vector512<double> oddReal = Unsafe.ReadUnaligned<Vector512<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, oddIndex)));
                    Vector512<double> oddImaginary = Unsafe.ReadUnaligned<Vector512<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)));

                    Vector512<double> twiddleReal = Unsafe.Add(ref realRootVectorBase, v);
                    Vector512<double> twiddleImaginary = Unsafe.Add(ref imaginaryRootVectorBase, v);

                    Vector512<double> tempReal = twiddleReal * oddReal - twiddleImaginary * oddImaginary;
                    Vector512<double> tempImaginary = twiddleReal * oddImaginary + twiddleImaginary * oddReal;

                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, evenIndex)),
                        evenReal + tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)),
                        evenImaginary + tempImaginary);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, oddIndex)),
                        evenReal - tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)),
                        evenImaginary - tempImaginary);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ProcessStageVector256(
            ref double realBase,
            ref double imaginaryBase,
            int n,
            int stage,
            double[] realRoots,
            double[] imaginaryRoots)
        {
            int blockSize = 1 << stage;
            int halfBlockSize = blockSize >> 1;
            const int VectorSize = 4;

            // View roots as Vector views
            ref Vector256<double> realRootVectorBase = ref Unsafe.As<double, Vector256<double>>(
                ref MemoryMarshal.GetArrayDataReference(realRoots));
            ref Vector256<double> imaginaryRootVectorBase = ref Unsafe.As<double, Vector256<double>>(
                ref MemoryMarshal.GetArrayDataReference(imaginaryRoots));

            int vectorizedOperations = halfBlockSize / VectorSize;

            for (int blockStart = 0; blockStart < n; blockStart += blockSize)
            {
                for (int v = 0; v < vectorizedOperations; v++)
                {
                    int offset = v * VectorSize;
                    int evenIndex = blockStart + offset;
                    int oddIndex = evenIndex + halfBlockSize;

                    // Load vectors (assuming aligned or hardware supports unaligned loads)
                    Vector256<double> evenReal = Unsafe.ReadUnaligned<Vector256<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, evenIndex)));
                    Vector256<double> evenImaginary = Unsafe.ReadUnaligned<Vector256<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)));
                    Vector256<double> oddReal = Unsafe.ReadUnaligned<Vector256<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, oddIndex)));
                    Vector256<double> oddImaginary = Unsafe.ReadUnaligned<Vector256<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)));

                    Vector256<double> twiddleReal = Unsafe.Add(ref realRootVectorBase, v);
                    Vector256<double> twiddleImaginary = Unsafe.Add(ref imaginaryRootVectorBase, v);

                    // Compute temp = odd * twiddle
                    // JIT typically generates FMA instructions for multiply-add patterns
                    Vector256<double> tempReal = twiddleReal * oddReal - twiddleImaginary * oddImaginary;
                    Vector256<double> tempImaginary = twiddleReal * oddImaginary + twiddleImaginary * oddReal;

                    // Store back
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, evenIndex)),
                        evenReal + tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)),
                        evenImaginary + tempImaginary);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, oddIndex)),
                        evenReal - tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)),
                        evenImaginary - tempImaginary);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ProcessStageVector128(
            ref double realBase,
            ref double imaginaryBase,
            int n,
            int stage,
            double[] realRoots,
            double[] imaginaryRoots)
        {
            int blockSize = 1 << stage;
            int halfBlockSize = blockSize >> 1;
            const int VectorSize = 2;

            ref Vector128<double> realRootVectorBase = ref Unsafe.As<double, Vector128<double>>(
                ref MemoryMarshal.GetArrayDataReference(realRoots));
            ref Vector128<double> imaginaryRootVectorBase = ref Unsafe.As<double, Vector128<double>>(
                ref MemoryMarshal.GetArrayDataReference(imaginaryRoots));

            int vectorizedOperations = halfBlockSize / VectorSize;

            for (int blockStart = 0; blockStart < n; blockStart += blockSize)
            {
                for (int v = 0; v < vectorizedOperations; v++)
                {
                    int evenIndex = blockStart + v * VectorSize;
                    int oddIndex = evenIndex + halfBlockSize;

                    Vector128<double> evenReal = Unsafe.ReadUnaligned<Vector128<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, evenIndex)));
                    Vector128<double> evenImaginary = Unsafe.ReadUnaligned<Vector128<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)));
                    Vector128<double> oddReal = Unsafe.ReadUnaligned<Vector128<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, oddIndex)));
                    Vector128<double> oddImaginary = Unsafe.ReadUnaligned<Vector128<double>>(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)));

                    Vector128<double> twiddleReal = Unsafe.Add(ref realRootVectorBase, v);
                    Vector128<double> twiddleImaginary = Unsafe.Add(ref imaginaryRootVectorBase, v);

                    Vector128<double> tempReal = twiddleReal * oddReal - twiddleImaginary * oddImaginary;
                    Vector128<double> tempImaginary = twiddleReal * oddImaginary + twiddleImaginary * oddReal;

                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, evenIndex)),
                        evenReal + tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)),
                        evenImaginary + tempImaginary);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref realBase, oddIndex)),
                        evenReal - tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<double, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)),
                        evenImaginary - tempImaginary);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ProcessStageScalar(
            ref double realBase,
            ref double imaginaryBase,
            int n,
            int stage,
            double[] realRoots,
            double[] imaginaryRoots)
        {
            int blockSize = 1 << stage;
            int halfBlockSize = blockSize >> 1;

            ref double realRootBase = ref MemoryMarshal.GetArrayDataReference(realRoots);
            ref double imaginaryRootBase = ref MemoryMarshal.GetArrayDataReference(imaginaryRoots);

            for (int blockStart = 0; blockStart < n; blockStart += blockSize)
            {
                int evenIndex = blockStart;
                int oddIndex = blockStart + halfBlockSize;

                double oddReal = Unsafe.Add(ref realBase, oddIndex);
                double oddImaginary = Unsafe.Add(ref imaginaryBase, oddIndex);
                double evenReal = Unsafe.Add(ref realBase, evenIndex);
                double evenImaginary = Unsafe.Add(ref imaginaryBase, evenIndex);

                Unsafe.Add(ref realBase, evenIndex) = evenReal + oddReal;
                Unsafe.Add(ref imaginaryBase, evenIndex) = evenImaginary + oddImaginary;
                Unsafe.Add(ref realBase, oddIndex) = evenReal - oddReal;
                Unsafe.Add(ref imaginaryBase, oddIndex) = evenImaginary - oddImaginary;

                // j > 0
                for (int j = 1; j < halfBlockSize; j++)
                {
                    evenIndex = blockStart + j;
                    oddIndex = evenIndex + halfBlockSize;

                    // Use Unsafe to skip bounds checking
                    oddReal = Unsafe.Add(ref realBase, oddIndex);
                    oddImaginary = Unsafe.Add(ref imaginaryBase, oddIndex);
                    evenReal = Unsafe.Add(ref realBase, evenIndex);
                    evenImaginary = Unsafe.Add(ref imaginaryBase, evenIndex);

                    double twiddleReal = Unsafe.Add(ref realRootBase, j);
                    double twiddleImaginary = Unsafe.Add(ref imaginaryRootBase, j);

                    double tempReal = Math.FusedMultiplyAdd(twiddleReal, oddReal, -twiddleImaginary * oddImaginary);
                    double tempImaginary = Math.FusedMultiplyAdd(twiddleReal, oddImaginary, twiddleImaginary * oddReal);

                    Unsafe.Add(ref realBase, evenIndex) = evenReal + tempReal;
                    Unsafe.Add(ref imaginaryBase, evenIndex) = evenImaginary + tempImaginary;
                    Unsafe.Add(ref realBase, oddIndex) = evenReal - tempReal;
                    Unsafe.Add(ref imaginaryBase, oddIndex) = evenImaginary - tempImaginary;
                }
            }
        }

        // =========================================================================================
        // Helper functions: Combined copy and bit reversal
        // =========================================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DeinterleaveAndBitReverse(
            Span<Complex> source,
            Span<double> destinationReal,
            Span<double> destinationImaginary)
        {
            int n = source.Length;
            int shift = 32 - BitOperations.Log2((uint)n);

            for (int i = 0; i < n; i++)
            {
                int reversedIndex = (int)(ReverseBits32((uint)i) >> shift);
                destinationReal[reversedIndex] = source[i].Real;
                destinationImaginary[reversedIndex] = source[i].Imaginary;
            }
        }

        /// <summary>
        /// 32-bit integer bit reversal algorithm (parallel swap method).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint ReverseBits32(uint n)
        {
            n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
            n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
            n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
            n = ((n >> 8) & 0x00FF00FF) | ((n & 0x00FF00FF) << 8);
            return (n >> 16) | (n << 16);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void InterleaveAndWriteBack(
            Span<double> sourceReal,
            Span<double> sourceImaginary,
            Span<Complex> destination)
        {
            ref double realBase = ref MemoryMarshal.GetReference(sourceReal);
            ref double imaginaryBase = ref MemoryMarshal.GetReference(sourceImaginary);
            ref Complex destinationBase = ref MemoryMarshal.GetReference(destination);

            for (int i = 0; i < destination.Length; i++)
            {
                Unsafe.Add(ref destinationBase, i) = new Complex(
                    Unsafe.Add(ref realBase, i),
                    Unsafe.Add(ref imaginaryBase, i));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void InterleaveAndWriteBackWithScale(
            Span<double> sourceReal,
            Span<double> sourceImaginary,
            Span<Complex> destination,
            double scale)
        {
            ref double realBase = ref MemoryMarshal.GetReference(sourceReal);
            ref double imaginaryBase = ref MemoryMarshal.GetReference(sourceImaginary);
            ref Complex destinationBase = ref MemoryMarshal.GetReference(destination);

            for (int i = 0; i < destination.Length; i++)
            {
                double scaledReal = Unsafe.Add(ref realBase, i) * scale;
                double scaledImaginary = Unsafe.Add(ref imaginaryBase, i) * scale;
                Unsafe.Add(ref destinationBase, i) = new Complex(scaledReal, scaledImaginary);
            }
        }
    }
}
