using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;
using System.Linq;

namespace NeuralNetwork.Layers
{
    internal class InputStandardizingLayer : ILayer, IEquatable<InputStandardizingLayer>
    {
        public double[] Mean { get; }
        public double[] StdDev { get; }
        public Matrix<double> MeanMatrix { get; private set; }
        public Matrix<double> Ones { get; private set; }
        public Matrix<double> StdDevMatrix { get; private set; }
        public Matrix<double> InputMinusMean { get; private set; }
        public Matrix<double> StandardizedInput { get; private set; }

        public InputStandardizingLayer(double[] mean, double[] stddev, ILayer underlyingLayer)
        {
            Mean = mean ?? throw new ArgumentNullException(nameof(mean));
            StdDev = stddev ?? throw new ArgumentNullException(nameof(stddev));
            UnderlyingLayer = underlyingLayer ?? throw new ArgumentNullException(nameof(underlyingLayer));
            if (mean.Length != stddev.Length || mean.Length != underlyingLayer.InputSize)
            {
                throw new ArgumentException("Mismatch between lengths of mean, stddev and input");
            }
            UpdateHelperMatrices(UnderlyingLayer.BatchSize);
        }

        public ILayer UnderlyingLayer { get; }

        public int LayerSize => UnderlyingLayer.LayerSize;

        public int InputSize => UnderlyingLayer.InputSize;

        public int BatchSize
        {
            get => UnderlyingLayer.BatchSize;
            set
            {
                UnderlyingLayer.BatchSize = value;
                UpdateHelperMatrices(value);
            }
        }

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;

        private void UpdateHelperMatrices(int batchSize)
        {
            var inputLength = Mean.Length;
            Ones = Matrix<double>.Build.Dense(1, batchSize, 1);
            MeanMatrix = Matrix<double>.Build.Dense(inputLength, batchSize);
            StdDevMatrix = Matrix<double>.Build.Dense(inputLength, batchSize);
            var tmpMean = Matrix<double>.Build.DenseOfColumns(new double[][] { Mean });
            tmpMean.Multiply(Ones, MeanMatrix);
            var tmpStdDev = Matrix<double>.Build.DenseOfColumns(new double[][] { StdDev });
            tmpStdDev.Multiply(Ones, StdDevMatrix);
            InputMinusMean = Matrix<double>.Build.Dense(inputLength, batchSize);
            StandardizedInput = Matrix<double>.Build.Dense(inputLength, batchSize);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);
        }

        public bool Equals(ILayer other)
        {
            return other is InputStandardizingLayer tstOther && Equals(tstOther);
        }

        public bool Equals(InputStandardizingLayer other)
        {
            return Enumerable.SequenceEqual(Mean, other.Mean) &&
                Enumerable.SequenceEqual(StdDev, other.StdDev) &&
                UnderlyingLayer.Equals(other.UnderlyingLayer);
        }

        public void Propagate(Matrix<double> input)
        {
            Standardize(input);
            UnderlyingLayer.Propagate(StandardizedInput);
        }

        public void UpdateParameters()
        {
            UnderlyingLayer.UpdateParameters();
        }

        private void Standardize(Matrix<double> input)
        {
            input.Subtract(MeanMatrix, InputMinusMean);
            InputMinusMean.PointwiseDivide(StdDevMatrix, StandardizedInput);
        }
    }
}