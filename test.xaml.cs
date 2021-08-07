using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Leadtools;
using RestSharp;
namespace WpfApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog openFileDialog = new Microsoft.Win32.OpenFileDialog();
            openFileDialog.Multiselect = true;
            openFileDialog.Filter = "DICOM files (*.dcm)|*.dcm";
            openFileDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            if (openFileDialog.ShowDialog() == true)
            {
                RestClient restClient = new RestClient("http://192.168.0.4/");
                RestRequest restRequest = new RestRequest("/uploader");
                //restRequest.RequestFormat = DataFormat.Json;
                restRequest.Method = Method.POST;
                restRequest.AddHeader("Authorization", "Authorization");
                restRequest.AddHeader("Content-Type", "multipart/form-data");
                //restRequest.AddFile("content", openFileDialog.FileName);
                restRequest.AddFile("file", "C:\\Users\\jihwa\\Desktop\\newjihwan\\1.jpeg");
                var response = restClient.Execute(restRequest);
            }
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {

            //Image myImage3 = new Image();
            BitmapImage bi3 = new BitmapImage();
            bi3.BeginInit();
            bi3.UriSource = new Uri("http://192.168.0.4/static/1.jpeg", UriKind.RelativeOrAbsolute);
            bi3.EndInit();
            image.Source = bi3;

            BitmapImage bi2 = new BitmapImage();
            bi2.BeginInit();
            bi2.UriSource = new Uri("http://192.168.0.4/static/2.jpeg", UriKind.RelativeOrAbsolute);
            bi2.EndInit();
            image1.Source = bi2;
        }

        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            image1.Opacity = Slider.Value;
        }

        private void Button_Click_2(object sender, RoutedEventArgs e)
        {
            
        }
    }
}
