using BenchmarkDotNet.Running;
using System;
using System.Numerics;
using Transform;

namespace Test
{
    internal class Program
    {
        static void Main(string[] args)
        {
            MathNet.Numerics.Control.TryUseNativeMKL();
            var summary = BenchmarkRunner.Run<Benchmark>();

            //const int inputSize = 8;
            //var cd = new Complex[inputSize];
            //var rd = new float[inputSize];

            //for (int i = 0; i < inputSize; i++)
            //{
            //    cd[i] = new Complex(i, i);
            //    rd[i] = i;
            //}
            //VortexFFT.Fft(cd);
            //VortexFFT.InverseFft(cd);
            //MathNet.Numerics.IntegralTransforms.Fourier.Forward(cd);
            //MathNet.Numerics.IntegralTransforms.Fourier.Inverse(cd);
        }
    }
}
