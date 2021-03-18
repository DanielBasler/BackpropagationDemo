using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;

namespace BackpropagationDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        NeuralNetwork neuralNetwork = null;
        bool SimulationStarted = false;
        double[,] inputs = new double[1022, 8];
        double[,] outputs = new double[1022, 1];
        int timerCount = 0;
        string[] CurrentOutputValue;
        int CurrentCount = 0;
        DispatcherTimer timer = new DispatcherTimer();
        public MainWindow()
        {
            InitializeComponent();
            InitControlButton();
            InitializeNeuralNetwork();
            timer.Tick += timer_Tick;
        }        

        int[] valueRange = new int[]{2,     3,     5,     7,    11,    13,    17,    19,    23,    29,    31,    37,    41,    43,
                               47,    53,    59,    61,    67,    71,    73,    79,    83,    89,    97,   101,   103,   107,
                              109,   113,   127,   131,   137,   139,   149,   151,   157,   163,   167,   173,   179,   181,
                              191,   193,   197,   199,   211,   223,   227,   229,   233,   239,   241,   251};


        private void InitControlButton()
        {
            bSimulation.IsEnabled = false;
            bTest.IsEnabled = false;            
            teNoMatch.IsReadOnly = true;
            teSearchedNumber.IsEnabled = false;
        }
        private void InitializeNeuralNetwork()
        {
            for (int i = 0; i < inputs.GetLength(0); i++)
            {                
                int num = i;
                int mask = 0x200;
                for (int j = 0; j < 8; j++)
                {
                    if ((num & mask) > 0)
                        inputs[i, j] = 1;
                    else
                        inputs[i, j] = 0;

                    mask = mask >> 1;
                }

                if (Array.BinarySearch(valueRange, i) >= 0)
                {                    
                    outputs[i, 0] = 1;
                }
                else
                {
                    outputs[i, 0] = 0;
                }
            }
        }

        private void BTraining_Click(object sender, RoutedEventArgs e)
        {            
            TraningStartNeuralNetwork();
            bSimulation.IsEnabled = true;
            bTest.IsEnabled = true;            
            teSearchedNumber.IsEnabled = true;
            StatusText.Text = "Training beendet";
            bTraining.IsEnabled = false;
        }

        private void TraningStartNeuralNetwork()
        {
            int numberOfHiddenNeurons = 10;
            int numberOfEpoche = 1022;
            neuralNetwork = new NeuralNetwork(inputs.GetLength(1), numberOfHiddenNeurons, (int)outputs.GetLength(1));            
            neuralNetwork.FirstTimeSettings();

            neuralNetwork.TrainingDataToNetwork(inputs, outputs, numberOfEpoche);
        }

        private void BCancel_Click(object sender, RoutedEventArgs e)
        {
            Application.Current.Shutdown();
        }

        private void BSimulation_Click(object sender, RoutedEventArgs e)
        {
            StartTimedSimulation();
        }

        private void StartTimedSimulation()
        {
            timerCount = 0;            
            teNoMatch.Text = "0";
            timer.Interval = TimeSpan.FromMilliseconds(250);
            timer.Start();
            SimulationStarted = true;
        }
        private void timer_Tick(object sender, EventArgs e)
        {
            CurrentCount = timerCount;
            CurrentOutputValue = neuralNetwork.TestDrive(CurrentCount, inputs).Split(new char[] { ' ' });

            DoStats(CurrentCount, Convert.ToInt32(CurrentOutputValue[0]));

            timerCount++;
            if (CurrentCount >= inputs.GetUpperBound(0))
            {
                VisualizingValues();                
                SimulationStarted = false;
                timer.Stop();
            }
            else
            {
                VisualizingValues();
            }

        }

        private void VisualizingValues()
        {            
            EpocheText.Text = "Epoche = " + CurrentCount.ToString();

            NeuronX1.Text = inputs[CurrentCount, 0].ToString();
            NeuronX2.Text = inputs[CurrentCount, 1].ToString();
            NeuronX3.Text = inputs[CurrentCount, 2].ToString();
            NeuronX4.Text = inputs[CurrentCount, 3].ToString();
            NeuronX5.Text = inputs[CurrentCount, 4].ToString();
            NeuronX6.Text = inputs[CurrentCount, 5].ToString();
            NeuronX7.Text = inputs[CurrentCount, 6].ToString();
            NeuronX8.Text = inputs[CurrentCount, 7].ToString();

            for(int i = 0; i < 1; i++)
            {
                if (SimulationStarted)
                {
                    OutputNeuron.Foreground = new SolidColorBrush(Colors.Green);
                    OutputNeuron.Text = CurrentOutputValue[i].ToString();
                }
                else
                {
                    OutputNeuron.Foreground = new SolidColorBrush(Colors.Black);
                    OutputNeuron.Text = outputs[CurrentCount, i].ToString();
                }
            }
        }

        private void DoStats(int currentCount, int v)
        {
            if (Array.BinarySearch(valueRange, currentCount) >= 0)
            {                
                if (v == 0)
                {                    
                    int val = Convert.ToInt32(teNoMatch.Text);
                    teNoMatch.Text = (val + 1).ToString();
                }
            }            
        }

        private void BTest_Click(object sender, RoutedEventArgs e)
        {
            CurrentCount = Convert.ToInt32(teSearchedNumber.Text);
            CurrentOutputValue = neuralNetwork.TestDrive(CurrentCount, inputs).Split(new char[] { ' ' });
            VisualizingValues();
        }
    }
}
