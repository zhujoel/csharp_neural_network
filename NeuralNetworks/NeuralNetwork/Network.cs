﻿using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork
{
    public sealed class Network : INetwork
    {
        public int BatchSize { get; set; }
        public int LayerNb { get; }
        internal ILayer OutputLayer => Layers[LayerNb - 1];
        public Matrix<double> Output => OutputLayer.Activation;
        

        public ILayer[] Layers { get; }

        public Mode Mode { get; set; }

        public Network(ILayer[] layers, int batchSize)
        {
            Layers = layers ?? throw new ArgumentNullException(nameof(layers));
            LayerNb = Layers.Length;
            if (LayerNb == 0)
            {
                throw new InvalidOperationException("The network must contain at least one layer");
            }
            BatchSize = batchSize;
        }

        public void Propagate(Matrix<double> input)
        {
            Layers[0].Propagate(input);
            for (int i = 1; i < LayerNb; i++)
            {
                Layers[i].Propagate(Layers[i - 1].Activation);
            }
        }

        public void Learn(Matrix<double> lossFunctionGradient)
        {
            BackpropAndUpdate(OutputLayer, lossFunctionGradient);
            for (int i = LayerNb - 2; i >= 0; i--)
            {
                BackpropAndUpdate(Layers[i], Layers[i + 1].WeightedError);
            }
        }

        private void BackpropAndUpdate(ILayer layer, Matrix<double> outputLayerError)
        {
            layer.BackPropagate(outputLayerError);
            layer.UpdateParameters();
        }
    }
}