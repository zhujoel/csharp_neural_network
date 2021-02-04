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
        public MomentumParameters MomentumParameter { get; set; }
        public Matrix<double> Zeta { get; set; }
        public Matrix<double> B_Rond { get; set; }
        public Matrix<double> Alpha { get; set; }
        public Matrix<double> Grad_Weight { get; set; }
        public Matrix<double> Grad_Bias { get; set; }
        public Matrix<double> Mat_Un { get; set; } // matrice de un (summary page 7 cours 3)

        // momentum
        public Matrix<double> v1; // weights
        public Matrix<double> v2; // bias

        public MomentumLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, MomentumParameters momentum)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);

            // attributs
            this.Mat_Un = Matrix<double>.Build.Dense(this.BatchSize, 1);
            Mat_Un.Multiply(0, Mat_Un);
            Mat_Un.Add(1, Mat_Un);

            this.Bias = bias.TransposeAndMultiply(Mat_Un);
            this.Weights = weights;
            this.Activator = activator;
            this.MomentumParameter = momentum;
            this.Grad_Weight = Matrix<double>.Build.Dense(weights.RowCount, weights.ColumnCount);
            this.Grad_Bias = Matrix<double>.Build.Dense(bias.RowCount, batchSize);

            this.v1 = this.Weights.Clone();
            this.v1.Multiply(0, this.v1);

            this.v2 = this.Bias.Clone();
            this.v2.Multiply(0, this.v2);
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
            this.B_Rond.Multiply(this.Mat_Un.Multiply(this.MomentumParameter.LearningRate / this.BatchSize)).TransposeAndMultiply(Mat_Un, this.Grad_Bias);
            this.Alpha.TransposeAndMultiply(this.B_Rond, this.Grad_Weight);

            this.v1.Multiply(this.MomentumParameter.Momentum, this.v1);
            this.v1.Subtract(Grad_Weight.Multiply(this.MomentumParameter.LearningRate/this.BatchSize), this.v1);
            
            this.v2.Multiply(this.MomentumParameter.Momentum, this.v2);
            this.v2.Subtract(Grad_Bias, this.v2);

            this.Weights.Add(v1, this.Weights);
            this.Bias.Add(v2, this.Bias);
        }
    }
}
