using BenchmarkDotNet.Attributes;
using MathNet.Numerics;
using System;
using System.Numerics;
using Transform;

namespace Test
{
    [MemoryDiagnoser]
    public class Benchmark
    {
        private Complex[] _sourceData;
        private Complex32[] _sourceData32;

        public int Iterations { get; set; } = 1000;

        [Params(4096, 8192)]
        public int N;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);

            _sourceData = new Complex[N];
            _sourceData32 = new Complex32[N];

            for (int i = 0; i < N; i++)
            {
                var real = rnd.NextDouble() * 2.0 - 1.0;
                var imag = rnd.NextDouble() * 2.0 - 1.0;
                _sourceData[i] = new Complex(real, imag);
                _sourceData32[i] = new Complex32((float)real, (float)imag);
            }
        }

        [Benchmark(Description = "Vortex FFT (double)")]
        public void Vortex_FFT()
        {
            var data = new Complex[N];
            Array.Copy(_sourceData, data, N);

            for (int i = 0; i < Iterations; i++)
            {
                VortexFFT.Fft(data);
            }

        }

        [Benchmark(Description = "Vortex FFT (float)")]
        public void Vortex_FFT32()
        {
            var data = new Complex32[N];
            Array.Copy(_sourceData32, data, N);

            for (int i = 0; i < Iterations; i++)
            {
                VortexFFT32.Fft(data);
            }
        }

        [Benchmark(Description = "MathNet FFT (double)")]
        public void MathNet_FFT()
        {
            var data = new Complex[N];
            Array.Copy(_sourceData, data, N);

            for (int i = 0; i < Iterations; i++)
            {
                MathNet.Numerics.IntegralTransforms.Fourier.Forward(data);
            }
        }

        [Benchmark(Description = "MathNet FFT (float)")]
        public void MathNet_FFT32()
        {
            var data = new Complex32[N];
            Array.Copy(_sourceData32, data, N);

            for (int i = 0; i < Iterations; i++)
            {
                MathNet.Numerics.IntegralTransforms.Fourier.Forward(data);
            }
        }
    }
}