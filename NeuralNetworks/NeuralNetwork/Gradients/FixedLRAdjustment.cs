namespace NeuralNetwork.Gradients
{
    internal  class FixedLRAdjustment : IGradientAdjustment
    {
        public double Adjust(double gradient)
        {
            // on doit retourner \eta * gradient
            return 0;
        }
    }
}
