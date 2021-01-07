using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Gradients
{
    internal class MomentumAdjustment : IGradientAdjustment
    {
        public double Adjust(double gradient)
        {
            // on doit retourner delta * v - \eta * g
            return 0;
        }
    }
}
