using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Gradients
{
    public class MomentumAdjustment
    {
        public static Matrix<double> Adjust(Matrix<double> gradient)
        {
            // on doit retourner delta * v - \eta * g
            return gradient.Multiply(5000);
        }
    }
}
