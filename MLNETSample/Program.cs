using Microsoft.ML;
using MLNETSample.Models;
using System;
using System.IO;

namespace MLNETSample
{
    class Program
    {
        const string ModelPath = "SentimentModel.zip";
        const string DefaultOutputColumnName = "Features";
        const string TrainingDataPath = @"Data\wikipedia-detox-250-line-data.tsv";
        const string TestDataPath = @"Data\wikipedia-detox-250-line-test.tsv";
        const string NewInputsPath = @"Data\inputs.txt";
        static void Main(string[] args)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Cyan;
            //1. Create our ML Context
            var mlContext = new MLContext();
            ITransformer model = null;

            //2. Load training data
            IDataView trainingDataview = mlContext.Data.LoadFromTextFile<SentimentData>(TrainingDataPath, hasHeader: true);


            if (File.Exists(ModelPath))
            {
                model = mlContext.Model.Load(ModelPath, out var modelInputSchema);
                Console.WriteLine($">>> Loaded model");
            }
            else
            {
                //3. Create and build the pipeline to prepare your data
                var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultOutputColumnName, inputColumnName: nameof(SentimentData.Text)).
                    Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

                //4. Get the model by training the pipeline that was build
                Console.WriteLine($">>> Creating and training the model");
                model = pipeline.Fit(trainingDataview);

                //5. Evaluate the model 
                Console.WriteLine($">>> Training completed");

                Console.WriteLine($">>> Evaluating the model with test data");
                IDataView testDataview = mlContext.Data.LoadFromTextFile<SentimentData>(TestDataPath, hasHeader: true);
                var predictions = model.Transform(testDataview);
                var results = mlContext.BinaryClassification.Evaluate(predictions);
                Console.WriteLine($">>> Model accuracy {results.Accuracy:P2}");
            }

            //6. Use Model - Create prediction engine related to the loaded trained model

            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            Console.WriteLine($">>> Created prediction engine based on model");

            string[] textInputs=File.ReadAllLines(NewInputsPath);

            Console.WriteLine($">>> Make predictions for new inputs");
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine("================START PREDICTIONS================");
            foreach (var input in textInputs)
            {
                var testInput = new SentimentData { Text = input };
                var predictionResult = predictionEngine.Predict(testInput);
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write("Prediction for ");
                Console.ForegroundColor = defaultColor;
                Console.Write($"\"{testInput.Text}\"");
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write(" is: ");

                if(Convert.ToBoolean(predictionResult.Prediction))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Toxic");
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("Non Toxic");
                }
              
            }

            //7. Save Model
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine("================END PREDICTIONS==================");
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($">>> Saved model");
            Console.ForegroundColor = defaultColor;
            mlContext.Model.Save(model, trainingDataview.Schema, ModelPath);

        }
    }
}
