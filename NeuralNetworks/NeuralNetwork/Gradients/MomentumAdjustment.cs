using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Gradients
{
    public class MomentumAdjustment : IGradientAdjustment
    {
        public double Momentum;
        public double LearningRate;

        readonly Matrix<double> Weight_Velocity;
        readonly Matrix<double> Bias_Velocity;


        double IGradientAdjustment.LearningRate => LearningRate;

        // we need those parameters to construct the velocities with the right dimensions
        public MomentumAdjustment(MomentumParameters momentum, Matrix<double> weights, Matrix<double> bias, int batchSize)
        {
            this.LearningRate = momentum.LearningRate;
            this.Momentum = momentum.Momentum;
            this.Weight_Velocity = Matrix<double>.Build.Dense(weights.RowCount, weights.ColumnCount);
            this.Bias_Velocity = Matrix<double>.Build.Dense(bias.RowCount, batchSize);

        }

        public void AdjustWeight(Matrix<double> weight, Matrix<double> gradient)
        {
            this.Weight_Velocity.Multiply(this.Momentum, this.Weight_Velocity);
            this.Weight_Velocity.Subtract(gradient.Multiply(this.LearningRate), this.Weight_Velocity);
            weight.Add(this.Weight_Velocity, weight);
        }

        public void AdjustBias(Matrix<double> bias, Matrix<double> gradient)
        {
            this.Bias_Velocity.Multiply(this.Momentum, this.Bias_Velocity);
            this.Bias_Velocity.Subtract(gradient, this.Bias_Velocity);
            bias.Add(this.Bias_Velocity, bias);
        }
    }
}
