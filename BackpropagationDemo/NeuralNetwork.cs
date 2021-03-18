using System;

namespace BackpropagationDemo
{
    public class NeuralNetwork
    {
        private int input;
        private int output;
        private int hidden;

        private double LearnHidden = 0.15f;
        private double LearnOutput = 0.2f;

        private double[] inputsToNetwork;
        private double[] desiredOutputs;

        private NetworkNode[] inputNN;
        private NetworkNode[] outputNN;
        private NetworkNode[] hiddenNN;
        
        public double error_compared_to_tolerance = 0;

        public NeuralNetwork(int i, int h, int o)
        {
            input = i;
            output = o;
            hidden = h;
            int ahm = 0;
            inputNN = new NetworkNode[input];
            outputNN = new NetworkNode[output];
            hiddenNN = new NetworkNode[hidden];
            Random rand = new Random(unchecked((int)DateTime.Now.Ticks));
            
            for (int x = 0; x < input; x++)
            {
                inputNN[x] = new NetworkNode();
                inputNN[x].weights = new double[hidden];
                for (int j = 0; j < hidden; j++)
                {
                    ahm = rand.Next() & 1;
                    inputNN[x].weights[j] = rand.NextDouble();
                    if (ahm == 0)
                        inputNN[x].weights[j] *= -1;
                }
            }

            for (int y = 0; y < hidden; y++)
            {
                hiddenNN[y] = new NetworkNode();
                hiddenNN[y].weights = new double[output];
                for (int j = 0; j < output; j++)
                {
                    hiddenNN[y].weights[j] = rand.NextDouble();
                }                    
            }

            for (int z = 0; z < output; z++)
            {
                outputNN[z] = new NetworkNode();
            }
        }

        public void FirstTimeSettings()
        {
            Random x = new Random(unchecked((int)DateTime.Now.Ticks));
            for (int i = 0; i < hidden; i++)
            {
                hiddenNN[i].Threshold = x.NextDouble();
            }
                
            for (int i = 0; i < output; i++)
            {
                outputNN[i].Threshold = x.NextDouble();
            }                
        }

        public void TrainingDataToNetwork(double[,] inputlist, double[,] outputlist, int iterations)
        {
            inputsToNetwork = new double[input];
            desiredOutputs = new double[output];

            int outputlistSampleLength = outputlist.GetUpperBound(0) + 1;
            int outputlistLength = outputlist.GetUpperBound(1) + 1;
            int inputlistLength = inputlist.GetUpperBound(1) + 1;            

            for (int i = 0; i < iterations; i++)
            {                
                for (int sampleindex = 0; sampleindex < outputlistSampleLength; sampleindex++)
                {
                    for (int j = 0; j < inputlistLength; j++)
                    {
                        inputsToNetwork[j] = inputlist[sampleindex, j];
                    }

                    for (int k = 0; k < outputlistLength; k++)
                    {
                        desiredOutputs[k] = outputlist[sampleindex, k];
                    }
                    
                    TrainingPattern();
                }
            }
        }

        private void TrainingPattern()
        {
            CalculateActivation();
            CalculateErrorOutput();
            CalculateErrorHidden();
            CalculateNewThresholds();
            CalculateNewWeightsInHidden();
            CalculateNewWeightsInInput();
        }       

        private void CalculateActivation()
        {
            int countHidden = 0;
            while (countHidden < hidden)
            {
                for (int ci = 0; ci < input; ci++)
                {
                    hiddenNN[countHidden].Activation += inputsToNetwork[ci] * inputNN[ci].weights[countHidden];
                }                   

                countHidden++;
            }         

            for (int x = 0; x < hidden; x++)
            {
                hiddenNN[x].Activation += hiddenNN[x].Threshold;
                hiddenNN[x].Activation = sigmoid(hiddenNN[x].Activation);
            }
            
            int countOutput = 0;
            while (countHidden < output)
            {
                for (int chi = 0; chi < hidden; chi++)
                {
                    outputNN[countOutput].Activation += hiddenNN[chi].Activation * hiddenNN[chi].weights[countOutput];
                }                    

                countOutput++;
            }
                        
            for (int x = 0; x < output; x++)
            {
                outputNN[x].Activation += outputNN[x].Threshold;
                outputNN[x].Activation = sigmoid(outputNN[x].Activation);
            }
        }

        private double sigmoid(double activation)
        {
            return 1 / (1 + Math.Exp(-activation));
        }

        private void CalculateErrorOutput()
        {
            for (int x = 0; x < output; x++)
            {
                outputNN[x].error = outputNN[x].Activation * (1 - outputNN[x].Activation) * (desiredOutputs[x] - outputNN[x].Activation);
            }                
        }

        private void CalculateErrorHidden()
        {
            int y = 0;
            while (y < hidden)
            {
                for (int x = 0; x < output; x++)
                {
                    hiddenNN[y].error += hiddenNN[y].weights[x] * outputNN[x].error;
                }
                hiddenNN[y].error *= hiddenNN[y].Activation * (1 - hiddenNN[y].Activation);
                y++;
            }
        }

        private void CalculateNewThresholds()
        {
            for (int x = 0; x < hidden; x++)
            {
                hiddenNN[x].Threshold += hiddenNN[x].error * LearnHidden;
            }

            for (int y = 0; y < output; y++)
            {
                outputNN[y].Threshold += outputNN[y].error * LearnOutput;
            }                
        }

        private void CalculateNewWeightsInHidden()
        {
            int x = 0;
            double temp = 0.0f;
            while (x < hidden)
            {
                temp = hiddenNN[x].Activation * LearnOutput;
                for (int y = 0; y < output; y++)
                {
                    hiddenNN[x].weights[y] += temp * outputNN[y].error;
                }

                x++;
            }
        }
        private void CalculateNewWeightsInInput()
        {
            int x = 0;
            double temp = 0.0f;
            while (x < input)
            {
                temp = inputsToNetwork[x] * LearnHidden;
                for (int y = 0; y < hidden; y++)
                {
                    inputNN[x].weights[y] += temp * hiddenNN[y].error;
                }

                x++;
            }
        }

        public string TestDrive(int index, double[,] inputlist)
        {
            PopulateInputList(inputlist, index);
            CalculateActivation();
            string value = GetValueOutput();            
            return value;
        }

        void PopulateInputList(double[,] inputlist, int index)
        {
            for (int j = 0; j < inputlist.GetUpperBound(1) + 1; j++)
            {
                inputsToNetwork[j] = inputlist[index, j];
            }
        }

        public string GetValueOutput()
        {
            string getOutput = "";
            
            for (int x = 0; x < output; x++)
            {
                if (outputNN[x].Activation > 0.5)
                    getOutput += "1" + " ";
                else
                    getOutput += "0" + " ";
            }
            return getOutput;

        }
    }
}
