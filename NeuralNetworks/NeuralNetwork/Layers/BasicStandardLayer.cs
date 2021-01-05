using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Layers
{
    internal class BasicStandardLayer : ILayer
    {
        public int LayerSize { get; }

        public int InputSize { get; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; }

        public Matrix<double> WeightedError { get; set; }

        // attributs perso
        public IActivator activator { get; }
        public Matrix<double> bias { get; set; }
        public Matrix<double> weights { get; set; }
        public Matrix<double> input { get; set; }
        public Matrix<double> outputIntermediaire { get; set; }
        public Matrix<double> bRond {get; set;}

        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);

            // algo 3 cours 1
            this.activator = activator;
            this.bias = bias;
            this.weights = weights;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            // algo 1 cours 2
            this.outputIntermediaire.Map(this.activator.ApplyDerivative, this.outputIntermediaire);
            this.outputIntermediaire.PointwiseMultiply(upstreamWeightedErrors, this.outputIntermediaire);
            this.bRond = outputIntermediaire;

            this.WeightedError = this.weights.Multiply(this.bRond);
        }

        public void Propagate(Matrix<double> input)
        {
            // algo 3 cours 1
            this.input = input;
            this.outputIntermediaire = weights.TransposeThisAndMultiply(input).Add(bias);
            this.outputIntermediaire.Map(this.activator.Apply, this.Activation);
        }

        // algo 2 cours 2
        public void UpdateParameters()
        {
            double learningRate = 1.0;

            var gradWeight = this.input.TransposeAndMultiply(this.bRond);
            var gradBias = this.bRond;

            this.weights.Subtract(gradWeight.Multiply(learningRate/this.BatchSize), this.weights);
            this.bias.Subtract(gradBias.Multiply(learningRate / this.BatchSize), this.bias);
        }
    }
}

// à opti affectation matrice