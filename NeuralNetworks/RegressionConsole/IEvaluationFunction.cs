namespace RegressionConsole
{
    internal interface IEvaluationFunction
    {
        double EvaluateError(double expectedOutput, double actualOutput);
    }
}