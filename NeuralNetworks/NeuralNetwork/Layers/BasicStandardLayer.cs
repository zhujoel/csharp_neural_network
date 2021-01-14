using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;

namespace NeuralNetwork.Layers
{
    internal class BasicStandardLayer : ILayer
    {
        public int LayerSize { get; }

        public int InputSize { get; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; }

        public Matrix<double> WeightedError { get; set; }

        // attributs supplémentaires
        public IActivator Activator { get; }
        public Matrix<double> Bias { get; set; }
        public Matrix<double> Weights { get; set; }
        public Matrix<double> Alpha { get; set; }
        public Matrix<double> Zeta { get; set; }
        public Matrix<double> BRond { get; set; }
        public FixedLearningRateParameters LearningParameter { get; set; }

        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, FixedLearningRateParameters learningParameter)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);

            // algo 3 cours 1
            this.Activator = activator;
            this.Bias = bias;
            this.Weights = weights;

            this.LearningParameter = learningParameter;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            // algo 1 cours 2
            this.Zeta.Map(this.Activator.ApplyDerivative, this.Zeta);
            this.BRond = this.Zeta.PointwiseMultiply(upstreamWeightedErrors);
            this.WeightedError = this.Weights.Multiply(this.BRond);
        }

        public void Propagate(Matrix<double> input)
        {
            // algo 3 cours 1
            this.Alpha = input;
            this.Zeta = this.Weights.TransposeThisAndMultiply(input).Add(Bias);
            this.Zeta.Map(this.Activator.Apply, this.Activation);
        }

        // algo 2 cours 2
        public void UpdateParameters()
        {
            var gradWeight = this.Alpha.TransposeAndMultiply(this.BRond);
            var gradBias = this.BRond;

            this.Weights.Subtract(gradWeight.Multiply(this.LearningParameter.LearningRate / this.BatchSize), this.Weights);
            this.Bias.Subtract(gradBias.Multiply(this.LearningParameter.LearningRate / this.BatchSize), this.Bias);
        }
    }
}

// échantillon de taille 100
// batch de taille 5

// on fait l'algo pour chaque batch de taille 5

// la matrice weight ne change pas 
// mettre à jour la taille de bias pour qu'elle ait la meme taile que le weight

