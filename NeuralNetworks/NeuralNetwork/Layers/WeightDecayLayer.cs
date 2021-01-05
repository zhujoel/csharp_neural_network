using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Layers
{
    internal class WeightDecayLayer : ILayer, IEquatable<WeightDecayLayer>
    {
        public WeightDecayLayer(ILayerWithStorage underlyingLayer, double decayRate)
        {
            UnderlyingLayer = underlyingLayer ?? throw new ArgumentNullException(nameof(underlyingLayer));
            DecayRate = decayRate;
            DecayedWeights = Matrix<double>.Build.Dense(UnderlyingLayer.MatrixStorage.Weights.RowCount, UnderlyingLayer.MatrixStorage.Weights.ColumnCount);
        }

        public Matrix<double> DecayedWeights { get; }
        public ILayerWithStorage UnderlyingLayer { get; }
        public double DecayRate { get; }

        public int LayerSize => UnderlyingLayer.LayerSize;

        public int InputSize => UnderlyingLayer.InputSize;

        public int BatchSize { get => UnderlyingLayer.BatchSize; set => UnderlyingLayer.BatchSize = value; }

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);
        }

        public bool Equals(ILayer other)
        {
            return other is WeightDecayLayer weight && Equals(weight);
        }

        public bool Equals(WeightDecayLayer other)
        {
            return UnderlyingLayer.Equals(other.UnderlyingLayer) && DecayRate == other.DecayRate;
        }

        public void Propagate(Matrix<double> input)
        {
            UnderlyingLayer.Propagate(input);
        }

        public void UpdateParameters()
        {
            UnderlyingLayer.MatrixStorage.Weights.Multiply(DecayRate, DecayedWeights);
            UnderlyingLayer.UpdateParameters();
            UnderlyingLayer.MatrixStorage.Weights.Subtract(DecayedWeights, UnderlyingLayer.MatrixStorage.Weights);
        }
    }
}