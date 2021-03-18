namespace BackpropagationDemo
{
    public class NetworkNode
    {
        public double[] weights;
        public double error;            

        public double Activation { set; get; }
        public double Threshold { set; get; }

        public NetworkNode()
        {
            Activation = 0;
            error = 0;
        }
    }
}
