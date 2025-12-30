using MathNet.Numerics;
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
    public static class VortexFFT32
    {
        // =========================================================================================
        // Pre-computed constants
        // =========================================================================================
        private const int MaxTableDepth = 20;
        private static readonly float[][] _realRootsOfUnity;
        private static readonly float[][] _imaginaryRootsOfUnity;
        private static readonly float[][] _imaginaryInverseRootsOfUnity;

        private static readonly int _simdWidth =
            Vector512.IsHardwareAccelerated ? 16 :
            Vector256.IsHardwareAccelerated ? 8 :
            Vector128.IsHardwareAccelerated ? 4 :
            0;

        static VortexFFT32()
        {
            _realRootsOfUnity = new float[MaxTableDepth][];
            _imaginaryRootsOfUnity = new float[MaxTableDepth][];
            _imaginaryInverseRootsOfUnity = new float[MaxTableDepth][];

            for (int i = 0; i < MaxTableDepth; i++)
            {
                int n = 1 << i;
                _realRootsOfUnity[i] = new float[n];
                _imaginaryRootsOfUnity[i] = new float[n];
                _imaginaryInverseRootsOfUnity[i] = new float[n];

                float angleStep = -2.0f * MathF.PI / n;
                for (int k = 0; k < n; k++)
                {
                    float angle = k * angleStep;
                    (float sin, float cos) = MathF.SinCos(angle);
                    _realRootsOfUnity[i][k] = cos;
                    _imaginaryRootsOfUnity[i][k] = sin;
                    _imaginaryInverseRootsOfUnity[i][k] = -sin;
                }
            }
        }

        /// <summary>
        /// Computes the Fast Fourier Transform (FFT) of a Complex32 array.
        /// </summary>
        public static void Fft(Complex32[] data) => Fft(data.AsSpan());

        /// <summary>
        /// Computes the Fast Fourier Transform (FFT) of a Complex32 span.
        /// </summary>
        public static void Fft(Span<Complex32> data)
        {
            int n = data.Length;
            if (n <= 1) return;
            if (!BitOperations.IsPow2(n))
                throw new ArgumentException("Data length must be a power of two.");

            // Smart memory allocation strategy
            float[]? poolArray = null;
            Span<float> buffer = n <= 8192
                ? stackalloc float[n * 2]
                : (poolArray = ArrayPool<float>.Shared.Rent(n * 2));

            try
            {
                Span<float> realParts = buffer[..n];
                Span<float> imaginaryParts = buffer.Slice(n, n);

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
                    ArrayPool<float>.Shared.Return(poolArray);
            }
        }

        /// <summary>
        /// Computes the Inverse Fast Fourier Transform (IFFT) of a Complex32 span.
        /// </summary>
        public static void InverseFft(Span<Complex32> data)
        {
            int n = data.Length;
            if (n == 0) return;

            float[]? poolArray = null;
            Span<float> buffer = n <= 8192
                ? stackalloc float[n * 2]
                : (poolArray = ArrayPool<float>.Shared.Rent(n * 2));

            try
            {
                Span<float> realParts = buffer[..n];
                Span<float> imaginaryParts = buffer.Slice(n, n);

                DeinterleaveAndBitReverse(data, realParts, imaginaryParts);

                FftCore(realParts, imaginaryParts, isForward: false);

                float scale = 1.0f / n;
                InterleaveAndWriteBackWithScale(realParts, imaginaryParts, data, scale);
            }
            finally
            {
                if (poolArray != null)
                    ArrayPool<float>.Shared.Return(poolArray);
            }
        }

        // =========================================================================================
        // Core FFT implementation
        // =========================================================================================

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void FftCore(Span<float> realParts, Span<float> imaginaryParts, bool isForward)
        {
            int n = realParts.Length;
            int logN = BitOperations.Log2((uint)n);

            var realRootsTable = _realRootsOfUnity;
            var imaginaryRootsTable = isForward ? _imaginaryRootsOfUnity : _imaginaryInverseRootsOfUnity;

            ref float realBase = ref MemoryMarshal.GetReference(realParts);
            ref float imaginaryBase = ref MemoryMarshal.GetReference(imaginaryParts);

            // Stage 1 & 2: scalar only
            if (logN >= 1)
                ProcessStageScalar(ref realBase, ref imaginaryBase, n, 1,
                    realRootsTable[1], imaginaryRootsTable[1]);
            if (logN >= 2)
                ProcessStageScalar(ref realBase, ref imaginaryBase, n, 2,
                    realRootsTable[2], imaginaryRootsTable[2]);

            // ----------------------------------------
            // Stage 3+
            // ----------------------------------------
            for (int stage = 3; stage <= logN; stage++)
            {
                int m2 = 1 << (stage - 1);

                if (_simdWidth == 16 && m2 >= 16)
                {
                    ProcessStageVector512(ref realBase, ref imaginaryBase, n, stage,
                        realRootsTable[stage], imaginaryRootsTable[stage]);
                }
                else if (_simdWidth == 8 && m2 >= 8)
                {
                    ProcessStageVector256(ref realBase, ref imaginaryBase, n, stage,
                        realRootsTable[stage], imaginaryRootsTable[stage]);
                }
                else if (_simdWidth == 4 && m2 >= 4)
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
            ref float realBase,
            ref float imaginaryBase,
            int n,
            int stage,
            float[] realRoots,
            float[] imaginaryRoots)
        {
            int blockSize = 1 << stage;
            int halfBlockSize = blockSize >> 1;
            const int VectorSize = 16;

            ref Vector512<float> realRootVectorBase = ref Unsafe.As<float, Vector512<float>>(
                ref MemoryMarshal.GetArrayDataReference(realRoots));
            ref Vector512<float> imaginaryRootVectorBase = ref Unsafe.As<float, Vector512<float>>(
                ref MemoryMarshal.GetArrayDataReference(imaginaryRoots));

            int vectorizedOperations = halfBlockSize / VectorSize;

            for (int blockStart = 0; blockStart < n; blockStart += blockSize)
            {
                for (int v = 0; v < vectorizedOperations; v++)
                {
                    int evenIndex = blockStart + v * VectorSize;
                    int oddIndex = evenIndex + halfBlockSize;

                    Vector512<float> evenReal = Unsafe.ReadUnaligned<Vector512<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, evenIndex)));
                    Vector512<float> evenImaginary = Unsafe.ReadUnaligned<Vector512<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)));
                    Vector512<float> oddReal = Unsafe.ReadUnaligned<Vector512<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, oddIndex)));
                    Vector512<float> oddImaginary = Unsafe.ReadUnaligned<Vector512<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)));

                    Vector512<float> twiddleReal = Unsafe.Add(ref realRootVectorBase, v);
                    Vector512<float> twiddleImaginary = Unsafe.Add(ref imaginaryRootVectorBase, v);

                    Vector512<float> tempReal = twiddleReal * oddReal - twiddleImaginary * oddImaginary;
                    Vector512<float> tempImaginary = twiddleReal * oddImaginary + twiddleImaginary * oddReal;

                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, evenIndex)),
                        evenReal + tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)),
                        evenImaginary + tempImaginary);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, oddIndex)),
                        evenReal - tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)),
                        evenImaginary - tempImaginary);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ProcessStageVector256(
            ref float realBase,
            ref float imaginaryBase,
            int n,
            int stage,
            float[] realRoots,
            float[] imaginaryRoots)
        {
            int blockSize = 1 << stage;
            int halfBlockSize = blockSize >> 1;
            const int VectorSize = 8;

            // View roots as Vector views
            ref Vector256<float> realRootVectorBase = ref Unsafe.As<float, Vector256<float>>(
                ref MemoryMarshal.GetArrayDataReference(realRoots));
            ref Vector256<float> imaginaryRootVectorBase = ref Unsafe.As<float, Vector256<float>>(
                ref MemoryMarshal.GetArrayDataReference(imaginaryRoots));

            int vectorizedOperations = halfBlockSize / VectorSize;

            for (int blockStart = 0; blockStart < n; blockStart += blockSize)
            {
                for (int v = 0; v < vectorizedOperations; v++)
                {
                    int offset = v * VectorSize;
                    int evenIndex = blockStart + offset;
                    int oddIndex = evenIndex + halfBlockSize;

                    Vector256<float> evenReal = Unsafe.ReadUnaligned<Vector256<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, evenIndex)));
                    Vector256<float> evenImaginary = Unsafe.ReadUnaligned<Vector256<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)));
                    Vector256<float> oddReal = Unsafe.ReadUnaligned<Vector256<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, oddIndex)));
                    Vector256<float> oddImaginary = Unsafe.ReadUnaligned<Vector256<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)));

                    Vector256<float> twiddleReal = Unsafe.Add(ref realRootVectorBase, v);
                    Vector256<float> twiddleImaginary = Unsafe.Add(ref imaginaryRootVectorBase, v);

                    Vector256<float> tempReal = twiddleReal * oddReal - twiddleImaginary * oddImaginary;
                    Vector256<float> tempImaginary = twiddleReal * oddImaginary + twiddleImaginary * oddReal;

                    // Store back
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, evenIndex)),
                        evenReal + tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)),
                        evenImaginary + tempImaginary);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, oddIndex)),
                        evenReal - tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)),
                        evenImaginary - tempImaginary);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ProcessStageVector128(
            ref float realBase,
            ref float imaginaryBase,
            int n,
            int stage,
            float[] realRoots,
            float[] imaginaryRoots)
        {
            int blockSize = 1 << stage;
            int halfBlockSize = blockSize >> 1;
            const int VectorSize = 4;

            ref Vector128<float> realRootVectorBase = ref Unsafe.As<float, Vector128<float>>(
                ref MemoryMarshal.GetArrayDataReference(realRoots));
            ref Vector128<float> imaginaryRootVectorBase = ref Unsafe.As<float, Vector128<float>>(
                ref MemoryMarshal.GetArrayDataReference(imaginaryRoots));

            int vectorizedOperations = halfBlockSize / VectorSize;

            for (int blockStart = 0; blockStart < n; blockStart += blockSize)
            {
                for (int v = 0; v < vectorizedOperations; v++)
                {
                    int evenIndex = blockStart + v * VectorSize;
                    int oddIndex = evenIndex + halfBlockSize;

                    Vector128<float> evenReal = Unsafe.ReadUnaligned<Vector128<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, evenIndex)));
                    Vector128<float> evenImaginary = Unsafe.ReadUnaligned<Vector128<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)));
                    Vector128<float> oddReal = Unsafe.ReadUnaligned<Vector128<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, oddIndex)));
                    Vector128<float> oddImaginary = Unsafe.ReadUnaligned<Vector128<float>>(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)));

                    Vector128<float> twiddleReal = Unsafe.Add(ref realRootVectorBase, v);
                    Vector128<float> twiddleImaginary = Unsafe.Add(ref imaginaryRootVectorBase, v);

                    Vector128<float> tempReal = twiddleReal * oddReal - twiddleImaginary * oddImaginary;
                    Vector128<float> tempImaginary = twiddleReal * oddImaginary + twiddleImaginary * oddReal;

                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, evenIndex)),
                        evenReal + tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, evenIndex)),
                        evenImaginary + tempImaginary);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref realBase, oddIndex)),
                        evenReal - tempReal);
                    Unsafe.WriteUnaligned(
                        ref Unsafe.As<float, byte>(ref Unsafe.Add(ref imaginaryBase, oddIndex)),
                        evenImaginary - tempImaginary);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ProcessStageScalar(
            ref float realBase,
            ref float imaginaryBase,
            int n,
            int stage,
            float[] realRoots,
            float[] imaginaryRoots)
        {
            int blockSize = 1 << stage;
            int halfBlockSize = blockSize >> 1;

            ref float realRootBase = ref MemoryMarshal.GetArrayDataReference(realRoots);
            ref float imaginaryRootBase = ref MemoryMarshal.GetArrayDataReference(imaginaryRoots);

            for (int blockStart = 0; blockStart < n; blockStart += blockSize)
            {
                int evenIndex = blockStart;
                int oddIndex = blockStart + halfBlockSize;

                float oddReal = Unsafe.Add(ref realBase, oddIndex);
                float oddImaginary = Unsafe.Add(ref imaginaryBase, oddIndex);
                float evenReal = Unsafe.Add(ref realBase, evenIndex);
                float evenImaginary = Unsafe.Add(ref imaginaryBase, evenIndex);

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

                    float twiddleReal = Unsafe.Add(ref realRootBase, j);
                    float twiddleImaginary = Unsafe.Add(ref imaginaryRootBase, j);

                    float tempReal = MathF.FusedMultiplyAdd(twiddleReal, oddReal, -twiddleImaginary * oddImaginary);
                    float tempImaginary = MathF.FusedMultiplyAdd(twiddleReal, oddImaginary, twiddleImaginary * oddReal);

                    Unsafe.Add(ref realBase, evenIndex) = evenReal + tempReal;
                    Unsafe.Add(ref imaginaryBase, evenIndex) = evenImaginary + tempImaginary;
                    Unsafe.Add(ref realBase, oddIndex) = evenReal - tempReal;
                    Unsafe.Add(ref imaginaryBase, oddIndex) = evenImaginary - tempImaginary;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DeinterleaveAndBitReverse(
            Span<Complex32> source,
            Span<float> destinationReal,
            Span<float> destinationImaginary)
        {
            int n = source.Length;
            int bits = BitOperations.Log2((uint)n);
            int shift = 32 - bits;

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
            Span<float> sourceReal,
            Span<float> sourceImaginary,
            Span<Complex32> destination)
        {
            ref float realBase = ref MemoryMarshal.GetReference(sourceReal);
            ref float imaginaryBase = ref MemoryMarshal.GetReference(sourceImaginary);
            ref Complex32 destinationBase = ref MemoryMarshal.GetReference(destination);

            for (int i = 0; i < destination.Length; i++)
            {
                Unsafe.Add(ref destinationBase, i) = new Complex32(
                    Unsafe.Add(ref realBase, i),
                    Unsafe.Add(ref imaginaryBase, i));
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void InterleaveAndWriteBackWithScale(
        Span<float> sourceReal,
        Span<float> sourceImaginary,
        Span<Complex32> destination,
        float scale)
        {
            ref float realBase = ref MemoryMarshal.GetReference(sourceReal);
            ref float imaginaryBase = ref MemoryMarshal.GetReference(sourceImaginary);
            ref Complex32 destinationBase = ref MemoryMarshal.GetReference(destination);

            for (int i = 0; i < destination.Length; i++)
            {
                float scaledReal = Unsafe.Add(ref realBase, i) * scale;
                float scaledImaginary = Unsafe.Add(ref imaginaryBase, i) * scale;
                Unsafe.Add(ref destinationBase, i) = new Complex32(scaledReal, scaledImaginary);
            }
        }
    }
}
