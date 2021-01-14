using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;

namespace NeuralNetwork.Layers
{
    class MomentumLayer : ILayer
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
        public MomentumParameters MomentumParameter { get; set; }

        // TODO: 
        // training steps (correspond au nb de fois que les gradients sont calculés) :
        // pour layer standard, si on a 4 couches avec 1000 neurones, on a 4000 steps
        // pour layer batch : on divise par la taille d'un batch, si la taille est 4 on a 1000 steps
        // TODO: vérifier que PropagationComparison est juste en calculant à la main

        public MomentumLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, MomentumParameters momentum)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);

            // algo 3 cours 1
            this.Activator = activator;
            this.Bias = bias;
            this.Weights = weights;
            this.MomentumParameter = momentum;
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
            // TODO: adjust gradients
            //this.Velocity = this.MomentumParameter.Momentum * Velocity - this.MomentumParameter.LearningRate * g;
            var gradWeight = this.Alpha.TransposeAndMultiply(this.BRond);
            var gradBias = this.BRond;

            // ici, on calcule les grad, puis on les adjust, puis on applique les lignes 78 et 79 (les 2 lignes suivantes) au gradient ajusté (les gradients ajusté sont teta + avec v la mesure d'ajustement)

            this.Weights.Subtract(gradWeight.Multiply(this.MomentumParameter.LearningRate/ this.BatchSize), this.Weights);
            this.Bias.Subtract(gradBias.Multiply(this.MomentumParameter.LearningRate / this.BatchSize), this.Bias);
        }
    }
}
