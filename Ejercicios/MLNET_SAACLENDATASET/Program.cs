using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNET_SAACLENDATASET
{
    class Program
    {
        static void Main()
        {
            const string fileInputPath = "data.csv";
            const string fileOutputPath = "data_preprocesed_ML.csv";

            var mlContext = new MLContext();


            TextLoader.Column[] columns =
            {
                new TextLoader.Column("ID", DataKind.Single, 0),
                new TextLoader.Column("Edad", DataKind.Single, 1),
                new TextLoader.Column("Genero", DataKind.String, 2),
                new TextLoader.Column("Ingresos_Mensuales", DataKind.Single, 3),
                new TextLoader.Column("Gastos_Anuales", DataKind.Single, 4),
                new TextLoader.Column("Educacion", DataKind.String, 5),
                new TextLoader.Column("Calificacion_Credito", DataKind.Single, 6),
                new TextLoader.Column("Tiempo_Empleo", DataKind.Single, 7)
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

            string GenreMode = rows.Where(r => !string.IsNullOrEmpty(r.Genero))
                                        .GroupBy(r => r.Genero)
                                        .OrderByDescending(g => g.Count())
                                        .First().Key;

            string EducationMode = rows.Where(r => !string.IsNullOrEmpty(r.Educacion))
                                        .GroupBy(r => r.Educacion)
                                        .OrderByDescending(g => g.Count())
                                        .First().Key;

            var categoricalMapping = mlContext.Transforms.CustomMapping<InputData, OutputData>((input, output) =>
                {
                    output.Genero = string.IsNullOrEmpty(input.Genero) ? GenreMode : input.Genero;
                    output.Educacion = string.IsNullOrEmpty(input.Educacion) ? EducationMode : input.Educacion;
                },
                contractName: "CustomMappingCategorical");

            var savingRateMapping = mlContext.Transforms.CustomMapping<InputSaving,OutputSaving>((input, output) =>
                {
                    output.Ratio_Ahorro = input.Ingresos_Mensuales * 12 / input.Gastos_Anuales;
                },
                contractName: "CustomMappingSavingRate");

            var pipeline = mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Edad", inputColumnName: "Edad", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean)
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Ingresos_Mensuales", inputColumnName: "Ingresos_Mensuales", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Gastos_Anuales", inputColumnName: "Gastos_Anuales", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Calificacion_Credito", inputColumnName: "Calificacion_Credito", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Tiempo_Empleo", inputColumnName: "Tiempo_Empleo", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(categoricalMapping)
                .Append(savingRateMapping)
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Edad_MinMax", inputColumnName: "Edad", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: "Ingresos_Mensuales_ZScore", inputColumnName: "Ingresos_Mensuales", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: "Gastos_Anuales_ZScore", inputColumnName: "Gastos_Anuales", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Calificacion_Credito_MinMax", inputColumnName: "Calificacion_Credito", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Tiempo_Empleo_MinMax", inputColumnName: "Tiempo_Empleo", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Ratio_Ahorro_MinMax", inputColumnName: "Ratio_Ahorro", fixZero: false))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Genero_OneHot", inputColumnName: "Genero"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Educacion_OneHot", inputColumnName: "Educacion"))
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

public class dataModel
{
    public float ID { get; set; } = float.NaN;
    public float Edad { get; set; } = float.NaN;
    public string Genero { get; set; } = string.Empty;
    public float Ingresos_Mensuales { get; set; } = float.NaN;
    public float Gastos_Anuales { get; set; } = float.NaN;
    public string Educacion { get; set; } = string.Empty;
    public float Calificacion_Credito { get; set; } = float.NaN;
    public float Tiempo_Empleo { get; set; } = float.NaN;
}

public class InputData
{
    public string Genero { get; set; } = string.Empty;    
    public string Educacion { get; set; } = string.Empty;
}

public class OutputData
{
    public string Genero { get; set; } = string.Empty;
    public string Educacion { get; set; } = string.Empty;
}

public class InputSaving
{
    public float Ingresos_Mensuales { get; set; } = float.NaN;
    public float Gastos_Anuales { get; set; } = float.NaN;
}

public class OutputSaving
{
    public float Ratio_Ahorro { get; set; } = float.NaN;
}