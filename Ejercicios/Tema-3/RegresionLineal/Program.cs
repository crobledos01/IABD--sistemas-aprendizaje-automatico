using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using RegresionLineal.Models;

namespace RegresionLineal
{
    public class EnergyPrediction
    {
        [ColumnName("Score")]
        public float Score { get; set; }
    }

    class Program
    {
        static void Main()
        {
            var mlContext = new MLContext(seed: 0);

            string dataPath = "./Data/energy_500.csv";

            var data = mlContext.Data.LoadFromTextFile<EnergyData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ','
            );

            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var pipeline =
                mlContext.Transforms.CopyColumns("Label", nameof(EnergyData.EnergyKWh))
                .Append(mlContext.Transforms.Concatenate("Features",
                    nameof(EnergyData.TempC),
                    nameof(EnergyData.HumidityPct),
                    nameof(EnergyData.PressureBar),
                    nameof(EnergyData.LoadPct),
                    nameof(EnergyData.Vibration)))
                .Append(mlContext.Regression.Trainers.Sdca());

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions);

            PrintMetrics(metrics);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<EnergyData, EnergyPrediction>(model);

            var testCases = new[]
            {
                new EnergyData{TempC=24,HumidityPct=48,PressureBar=1.01f,LoadPct=62,Vibration=2.5f},
                new EnergyData{TempC=30,HumidityPct=60,PressureBar=1.00f,LoadPct=80,Vibration=3.4f},
                new EnergyData{TempC=18,HumidityPct=40,PressureBar=1.02f,LoadPct=50,Vibration=2.0f}
            };

            Console.WriteLine("\n===== PREDICCIONES =====");
            Console.WriteLine($"{"TempC",10} {"Hum%",10} {"Press",10} {"Load%",10} {"Vib",10} {"Pred KWh",10}");
            Console.WriteLine(new string('-', 70));

            foreach (var d in testCases)
            {
                var result = predictionEngine.Predict(d);

                Console.WriteLine($"{d.TempC,10:F1} {d.HumidityPct,10:F1} {d.PressureBar,10:F3} {d.LoadPct,10:F1} {d.Vibration,10:F2} {result.Score,10:F2}");
            }

            Console.WriteLine(new string('-', 70));
        }

        static void PrintMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine("===== MÉTRICAS =====");
            Console.WriteLine($"MAE : {metrics.MeanAbsoluteError:F4}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F4}");
            Console.WriteLine($"MSE : {metrics.MeanSquaredError:F4}");
            Console.WriteLine($"R²  : {metrics.RSquared:F4}");
            Console.WriteLine($"loss: {metrics.LossFunction:F4}");
        }
    }
}