using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Layers
{
    internal class L2PenaltyLayer : ILayer
    {
        public int LayerSize => UnderlyingLayer.LayerSize;

        public int InputSize => UnderlyingLayer.InputSize;

        public int BatchSize
        {
            get => UnderlyingLayer.BatchSize;
            set => UnderlyingLayer.BatchSize = value;
        }

        public Matrix<double> Activation => UnderlyingLayer.Activation;

        public Matrix<double> WeightedError => UnderlyingLayer.WeightedError;
        // attributs supplémentaires
        public Matrix<double> PenaltyWeights { get; }
        public MomentumLayer UnderlyingLayer { get; }
        public double Kappa { get; }
        public L2PenaltyLayer(ILayer underlyingLayer, double penaltyCoefficient)
        {
            UnderlyingLayer = underlyingLayer as MomentumLayer;
            Kappa = penaltyCoefficient;
            PenaltyWeights = Matrix<double>.Build.Dense(UnderlyingLayer.Weights.RowCount, UnderlyingLayer.Weights.ColumnCount);
        }

        public void Propagate(Matrix<double> input)
        {
            UnderlyingLayer.Propagate(input);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);
            UnderlyingLayer.Weights.Multiply(Kappa, PenaltyWeights);
            UnderlyingLayer.Grad_Weight.Add(PenaltyWeights, UnderlyingLayer.Grad_Weight);
        }

        public void UpdateParameters()
        {
            UnderlyingLayer.UpdateParameters();
        }
    }
}