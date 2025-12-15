using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace meteorologia
{
    class Program
    {
        static void Main()
        {
            const string fileInputPath = "datos_meteorologicos.csv";
            const string fileOutputPath = "datos_procesados.csv";

            var mlContext = new MLContext();

            TextLoader.Column[] columns =
            {
                new TextLoader.Column("Fecha", DataKind.String, 0),
                new TextLoader.Column("Temperatura_C", DataKind.Single, 1),
                new TextLoader.Column("Humedad", DataKind.Single, 2),
                new TextLoader.Column("Tipo_de_Clima", DataKind.String, 3),
                new TextLoader.Column("Velocidad_Viento_kmh", DataKind.Single, 4),
                new TextLoader.Column("Precipitacion_mm", DataKind.Single, 5),
                new TextLoader.Column("Presion_hPa", DataKind.Single, 6),
            };

            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                HasHeader = true,
                Separators = new[] { ',' },
                //AllowQuoting = true,
                TrimWhitespace = true,
                MissingRealsAsNaNs = true,
                Columns = columns
            });

            IDataView data;
            data = loader.Load(fileInputPath);

            var rows = mlContext.Data.CreateEnumerable<dataModel>(data, false).ToList();

            var energyGeneratedMapping = mlContext.Transforms.CustomMapping<InputEnergy, OutputEnergy>((input, output) =>
                {
                    if (input.Temperatura_C < -10 || input.Precipitacion_mm > 10)
                    {
                        output.Energia_Generada = 0;
                    } else
                    {
                        output.Energia_Generada = (float)Math.Pow(input.Velocidad_Viento_kmh, 3);

                    }
                }, contractName: "CustomMappingEnergyGenerated");

            var pipeline = mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Temperatura_C", inputColumnName: "Temperatura_C", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean)
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Humedad", inputColumnName: "Humedad", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Velocidad_Viento_kmh", inputColumnName: "Velocidad_Viento_kmh", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Precipitacion_mm", inputColumnName: "Precipitacion_mm", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Presion_hPa", inputColumnName: "Presion_hPa", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(energyGeneratedMapping)
                .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: "Temperatura_C_ZScore", inputColumnName: "Temperatura_C", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Humedad_MinMax", inputColumnName: "Humedad", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Velocidad_Viento_kmh_MinMax", inputColumnName: "Velocidad_Viento_kmh", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Precipitacion_mm_MinMax", inputColumnName: "Precipitacion_mm", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Presion_hPa_MinMax", inputColumnName: "Presion_hPa", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Energia_Generada_MinMax", inputColumnName: "Energia_Generada", fixZero: false))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Fecha_OneHot", inputColumnName: "Fecha"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Tipo_de_Clima_OneHot", inputColumnName: "Tipo_de_Clima"))
                .Append(mlContext.Transforms.SelectColumns(["Features"]));
                
            var trasformer = pipeline.Fit(data);
            IDataView transformedData = trasformer.Transform(data);

            using (var fs = File.Create(fileOutputPath))
            {
                mlContext.Data.SaveAsText(data: transformedData, stream: fs, separatorChar: ',', headerRow: true, forceDense: false /*, schema: false */);
            }
            
        }
    }
}

public class InputEnergy
{
    public float Velocidad_Viento_kmh { get; set; } = float.NaN;
    public float Precipitacion_mm { get; set; } = float.NaN;
    public float Temperatura_C { get; set; } = float.NaN;
}
public class OutputEnergy
{
    public float Energia_Generada { get; set; } = float.NaN;
}
public class dataModel
{
    public string Fecha { get; set; } = string.Empty;
    public float Temperatura_C { get; set; } = float.NaN;
    public float Humedad { get; set; } = float.NaN;
    public string Tipo_de_Clima { get; set; } = string.Empty;
    public float Velocidad_Viento_kmh { get; set; } = float.NaN;
    public float Precipitacion_mm { get; set; } = float.NaN;
    public float Presion_hPa { get; set; } = float.NaN;
}