using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Gradients;

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
        public IGradientAdjustment Adjustment { get; set; }
        public Matrix<double> Zeta { get; set; }
        public Matrix<double> B_Rond { get; set; }
        public Matrix<double> Alpha { get; set; }
        public Matrix<double> Grad_Weight { get; set; }
        public Matrix<double> Grad_Bias { get; set; }
        public Matrix<double> Mat_Un { get; set; } // matrice de un (summary page 7 cours 3)


        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, IGradientAdjustment adjustment)
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
            this.Grad_Weight = Matrix<double>.Build.Dense(weights.RowCount, weights.ColumnCount);
            this.Grad_Bias = Matrix<double>.Build.Dense(bias.RowCount, batchSize);
            this.Adjustment = adjustment;
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
            this.B_Rond.Multiply(this.Mat_Un.Multiply(this.Adjustment.LearningRate)).TransposeAndMultiply(Mat_Un, this.Grad_Bias); // we take into account the dimension change from batch size
            this.Alpha.TransposeAndMultiply(this.B_Rond, this.Grad_Weight);

            this.Adjustment.AdjustWeight(this.Weights, Grad_Weight);
            this.Adjustment.AdjustBias(this.Bias, Grad_Bias);
        }
    }
}