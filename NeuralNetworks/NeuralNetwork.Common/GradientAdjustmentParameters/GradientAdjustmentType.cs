namespace NeuralNetwork.Common.GradientAdjustmentParameters
{
    /// <summary>
    /// Gradient adjustment types.
    /// </summary>
    public enum GradientAdjustmentType
    {
        FixedLearningRate,
        Adam,
        Momentum,
        Nesterov
    }
}