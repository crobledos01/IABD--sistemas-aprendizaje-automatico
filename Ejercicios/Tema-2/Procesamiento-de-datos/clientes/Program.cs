using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace meteorologia
{
    class Program
    {
        static void Main()
        {
            const string fileInputPath = "datos_clientes.csv";
            const string fileOutputPath = "datos_procesados.csv";

            var mlContext = new MLContext();

            TextLoader.Column[] columns =
            {
                new TextLoader.Column("ID_Cliente", DataKind.Single, 0),
                new TextLoader.Column("Edad", DataKind.Single, 1),
                new TextLoader.Column("Genero", DataKind.String, 2),
                new TextLoader.Column("Ingreso_Mensual_USD", DataKind.Single, 3),
                new TextLoader.Column("Producto_Preferido", DataKind.String, 4),
                new TextLoader.Column("Frecuencia_Compra_mensual", DataKind.Single, 5),
                new TextLoader.Column("Region", DataKind.String, 6),
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

            string ProductMode = rows.Where(r => !string.IsNullOrEmpty(r.Producto_Preferido))
                                        .GroupBy(r => r.Producto_Preferido)
                                        .OrderByDescending(g => g.Count())
                                        .First().Key;

            string RegionMode = rows.Where(r => !string.IsNullOrEmpty(r.Region))
                                        .GroupBy(r => r.Region)
                                        .OrderByDescending(g => g.Count())
                                        .First().Key;

            var categoricalMapping = mlContext.Transforms.CustomMapping<InputCategorical, OutputCategorical>((input, output) =>
                {
                    output.Genero = string.IsNullOrEmpty(input.Genero) ? GenreMode : input.Genero;
                    output.Producto_Preferido = string.IsNullOrEmpty(input.Producto_Preferido) ? ProductMode : input.Producto_Preferido;
                    output.Region = string.IsNullOrEmpty(input.Region) ? RegionMode : input.Region;
                },
                contractName: "CustomMappingCategorical");

            var RangoEdadMapping = mlContext.Transforms.CustomMapping<InputEnergy, OutputEnergy>((input, output) =>
                {
                    if(input.Edad < 27)
                    {
                        output.Rango_Edad = "Joven";
                    } else if(input.Edad >= 30 && input.Edad < 47)
                    {
                        output.Rango_Edad = "Adulto";
                    } else
                    {
                        output.Rango_Edad = "Senior";
                    }
                }, contractName: "CustomMappingRangoEdad");

            var pipeline = mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Edad", inputColumnName: "Edad", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean)
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Ingreso_Mensual_USD", inputColumnName: "Ingreso_Mensual_USD", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Frecuencia_Compra_mensual", inputColumnName: "Frecuencia_Compra_mensual", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mode))
                .Append(RangoEdadMapping)
                .Append(categoricalMapping)
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Edad_MinMax", inputColumnName: "Edad", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Ingreso_Mensual_USD_MinMax", inputColumnName: "Ingreso_Mensual_USD", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Frecuencia_Compra_mensual_MinMax", inputColumnName: "Frecuencia_Compra_mensual", fixZero: false))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Genero_OneHot", inputColumnName: "Genero"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Producto_Preferido_OneHot", inputColumnName: "Producto_Preferido"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Region_OneHot", inputColumnName: "Region"))
                .Append(mlContext.Transforms.Concatenate("Features", ["Edad_MinMax", "Genero_OneHot", "Ingreso_Mensual_USD_MinMax", "Producto_Preferido_OneHot", "Frecuencia_Compra_mensual_MinMax", "Region_OneHot"]))
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
    public float Edad { get; set; } = float.NaN;
}
public class OutputEnergy
{
    public string Rango_Edad { get; set; } = string.Empty;
}

public class InputCategorical
{
    public string Genero { get; set; } = string.Empty;
    public string Producto_Preferido { get; set; } = string.Empty;
    public string Region { get; set; } = string.Empty;
}
public class OutputCategorical
{
    public string Genero { get; set; } = string.Empty;
    public string Producto_Preferido { get; set; } = string.Empty;
    public string Region { get; set; } = string.Empty;
}
public class dataModel
{
    public float ID_Cliente { get; set; } = float.NaN;
    public float Edad { get; set; } = float.NaN;
    public string Genero { get; set; } = string.Empty;
    public float Ingreso_Mensual_USD { get; set; } = float.NaN;
    public string Producto_Preferido { get; set; } = string.Empty;
    public float Frecuencia_Compra_mensual { get; set; } = float.NaN;
    public string Region { get; set; } = string.Empty;
}