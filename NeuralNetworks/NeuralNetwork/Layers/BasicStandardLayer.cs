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
        public FixedLearningRateParameters LearningParameter { get; set; }
        public Matrix<double> Zeta { get; set; }
        public Matrix<double> B_Rond { get; set; }
        public Matrix<double> Alpha { get; set; }
        //public Matrix<double> Grad_Weight { get; set; } // L2
        public Matrix<double> Mat_Un { get; set; } // matrice de un (summary page 7 cours 3)


        public BasicStandardLayer(Matrix<double> weights, Matrix<double> bias, int batchSize, IActivator activator, FixedLearningRateParameters learningParameter)
        {
            BatchSize = batchSize;
            InputSize = weights.RowCount;
            LayerSize = weights.ColumnCount;
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);

            // attributs
            this.Bias = bias.Clone();
            for (int i = 0; i < this.BatchSize-1; ++i)
            {
                this.Bias = this.Bias.Append(bias);
            }

            this.Weights = weights;
            this.Activator = activator;
            this.LearningParameter = learningParameter;
            this.Alpha = Matrix<double>.Build.Dense(InputSize, batchSize);


            this.Mat_Un = Matrix<double>.Build.Dense(this.BatchSize, 1);
            Mat_Un.Multiply(0, Mat_Un);
            Mat_Un.Add(1, Mat_Un);


            // L2
            //this.Grad_Weight = weights;
            //this.Grad_Weight.Multiply(0, this.Grad_Weight);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            //this.Grad_Weight.Multiply(0, this.Grad_Weight); // L2
            this.B_Rond = this.Zeta.Map(this.Activator.ApplyDerivative).PointwiseMultiply(upstreamWeightedErrors);
            this.WeightedError = this.Weights.Multiply(this.B_Rond);
        }

        public void Propagate(Matrix<double> input)
        {
            input.CopyTo(this.Alpha);
            this.Zeta = this.Weights.TransposeThisAndMultiply(input).Add(Bias);
            this.Zeta.Map(this.Activator.Apply, this.Activation);
        }

        public void UpdateParameters()
        {
            var Grad_Bias = this.B_Rond.Multiply(this.Mat_Un);
            var Grad_Bias_Clone = Grad_Bias.Clone();
            for (int i = 0; i < this.BatchSize - 1; ++i)
            {
                Grad_Bias = Grad_Bias.Append(Grad_Bias_Clone);
            }

            //this.Grad_Weight.Add(this.Alpha.TransposeAndMultiply(this.B_Rond), this.Grad_Weight); // L2
            var Grad_Weight = this.Alpha.TransposeAndMultiply(this.B_Rond);

            this.Weights.Subtract(Grad_Weight.Multiply(this.LearningParameter.LearningRate/this.BatchSize), this.Weights);
            this.Bias.Subtract(Grad_Bias.Multiply(this.LearningParameter.LearningRate/this.BatchSize), this.Bias);
        }
    }
}