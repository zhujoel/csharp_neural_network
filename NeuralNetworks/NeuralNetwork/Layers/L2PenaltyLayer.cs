using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Layers
{
    internal class L2PenaltyLayer : ILayer, IEquatable<L2PenaltyLayer>
    {
        public L2PenaltyLayer(ILayerWithStorage underlyingLayer, double penaltyCoefficient)
        {
            UnderlyingLayer = underlyingLayer ?? throw new ArgumentNullException(nameof(underlyingLayer));
            PenaltyCoefficient = penaltyCoefficient;
            PenaltyWeights = Matrix<double>.Build.Dense(UnderlyingLayer.MatrixStorage.Weights.RowCount, UnderlyingLayer.MatrixStorage.Weights.ColumnCount);
        }

        public Matrix<double> PenaltyWeights { get; }
        public ILayerWithStorage UnderlyingLayer { get; }
        public double PenaltyCoefficient { get; }
        //public IMatrixStorage MatrixStorage => UnderlyingLayer.MatrixStorage;

        public int LayerSize => UnderlyingLayer.LayerSize;

        public int InputSize => UnderlyingLayer.InputSize;

        public int BatchSize
        {
            get => UnderlyingLayer.BatchSize;
            set => UnderlyingLayer.BatchSize = value;
        }

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);
            UnderlyingLayer.MatrixStorage.Weights.Multiply(PenaltyCoefficient, PenaltyWeights);
            UnderlyingLayer.MatrixStorage.WeightGradients.Add(PenaltyWeights, UnderlyingLayer.MatrixStorage.WeightGradients);
        }

        public bool Equals(ILayer other)
        {
            return other is L2PenaltyLayer penaltyLayer && Equals(penaltyLayer);
        }

        public bool Equals(L2PenaltyLayer other)
        {
            return UnderlyingLayer.Equals(other.UnderlyingLayer) && PenaltyCoefficient == other.PenaltyCoefficient;
        }

        public void Propagate(Matrix<double> input)
        {
            UnderlyingLayer.Propagate(input);
        }

        public void UpdateParameters()
        {
            UnderlyingLayer.UpdateParameters();
        }
    }
}