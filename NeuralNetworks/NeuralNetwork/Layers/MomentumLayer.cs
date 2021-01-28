using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Gradients;

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
        public MomentumParameters MomentumParameter { get; set; }
        public Matrix<double> Zeta { get; set; }
        public Matrix<double> B_Rond { get; set; }
        public Matrix<double> Alpha { get; set; }

        public MomentumLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, MomentumParameters momentum)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);

            // attributs
            this.Weights = weights;
            this.Bias = bias;
            this.Activator = activator;
            this.MomentumParameter = momentum;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            this.B_Rond = this.Zeta.Map(this.Activator.ApplyDerivative).PointwiseMultiply(upstreamWeightedErrors);
            this.WeightedError = this.Weights.Multiply(this.B_Rond);
        }

        public void Propagate(Matrix<double> input)
        {
            this.Alpha = input;
            this.Zeta = this.Weights.TransposeThisAndMultiply(input).Add(Bias);
            this.Zeta.Map(this.Activator.Apply, this.Activation);
        }

        public void UpdateParameters()
        {
            var Grad_Bias = this.B_Rond;
            var Grad_Weight = this.Alpha.TransposeAndMultiply(this.B_Rond);

            Grad_Bias = MomentumAdjustment.Adjust(Grad_Bias);

            this.Weights = this.Weights.Subtract(Grad_Weight.Multiply(this.MomentumParameter.LearningRate));
            this.Bias = this.Bias.Subtract(Grad_Bias.Multiply(this.MomentumParameter.LearningRate));
        }
    }
}
