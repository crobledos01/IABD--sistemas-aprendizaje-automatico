using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace meteorologia
{
    class Program
    {
        static void Main()
        {
            const string fileInputPath = "datos_vivienda_coste.csv";
            const string fileOutputPath = "datos_procesados.csv";

            var mlContext = new MLContext();

            TextLoader.Column[] columns =
            {
                new TextLoader.Column("ID_Vivienda", DataKind.Single, 0),
                new TextLoader.Column("Numero_de_Habitaciones", DataKind.Single, 1),
                new TextLoader.Column("Ubicacion", DataKind.String, 2),
                new TextLoader.Column("Tamaño_m2", DataKind.Single, 3),
                new TextLoader.Column("Tipo_de_Vivienda", DataKind.String, 4),
                new TextLoader.Column("Coste_USD", DataKind.Single, 5),
                new TextLoader.Column("Año_de_Construcción", DataKind.Single, 6),
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

            string UbicationMode = rows.Where(r => !string.IsNullOrEmpty(r.Ubicacion))
                                        .GroupBy(r => r.Ubicacion)
                                        .OrderByDescending(g => g.Count())
                                        .First().Key;

            string TypeMode = rows.Where(r => !string.IsNullOrEmpty(r.Tipo_de_Vivienda))
                                        .GroupBy(r => r.Tipo_de_Vivienda)
                                        .OrderByDescending(g => g.Count())
                                        .First().Key;

            var categoricalMapping = mlContext.Transforms.CustomMapping<InputCategorical, OutputCategorical>((input, output) =>
                {
                    output.Ubicacion = string.IsNullOrEmpty(input.Ubicacion) ? UbicationMode : input.Ubicacion;
                    output.Tipo_de_Vivienda = string.IsNullOrEmpty(input.Tipo_de_Vivienda) ? TypeMode : input.Tipo_de_Vivienda;
                },
                contractName: "CustomMappingCategorical");

            var PrecioM2Mapping = mlContext.Transforms.CustomMapping<InputEnergy, OutputEnergy>((input, output) =>
                {
                    output.Precio_m2 = input.Coste_USD / input.Tamaño_m2;
                }, contractName: "CustomMappingRangoEdad");

            var pipeline = mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Numero_de_Habitaciones", inputColumnName: "Numero_de_Habitaciones", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mode)
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Tamaño_m2", inputColumnName: "Tamaño_m2", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Coste_USD", inputColumnName: "Coste_USD", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: "Año_de_Construcción", inputColumnName: "Año_de_Construcción", replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mode))
                .Append(PrecioM2Mapping)
                .Append(categoricalMapping)
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Numero_de_Habitaciones_MinMax", inputColumnName: "Numero_de_Habitaciones", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Tamaño_m2_MinMax", inputColumnName: "Tamaño_m2", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Coste_USD_MinMax", inputColumnName: "Coste_USD", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Año_de_Construcción_MinMax", inputColumnName: "Año_de_Construcción", fixZero: false))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "Precio_m2_MinMax", inputColumnName: "Precio_m2", fixZero: false))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Ubicacion_OneHot", inputColumnName: "Ubicacion"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Tipo_de_Vivienda_OneHot", inputColumnName: "Tipo_de_Vivienda"))
                .Append(mlContext.Transforms.Concatenate("Features", ["Numero_de_Habitaciones_MinMax", "Tamaño_m2_MinMax", "Coste_USD_MinMax", "Año_de_Construcción_MinMax", "Precio_m2_MinMax", "Ubicacion_OneHot", "Tipo_de_Vivienda_OneHot"]))
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
    public float Tamaño_m2 { get; set; } = float.NaN;
    public float Coste_USD { get; set; } = float.NaN;
}
public class OutputEnergy
{
    public float Precio_m2 { get; set; } = float.NaN;
}

public class InputCategorical
{
    public string Ubicacion { get; set; } = string.Empty;
    public string Tipo_de_Vivienda { get; set; } = string.Empty;
}
public class OutputCategorical
{
    public string Ubicacion { get; set; } = string.Empty;
    public string Tipo_de_Vivienda { get; set; } = string.Empty;
}
public class dataModel
{
    public float ID_Vivienda { get; set; } = float.NaN;
    public float Numero_de_Habitaciones { get; set; } = float.NaN;
    public string Ubicacion { get; set; } = string.Empty;
    public float Tamaño_m2 { get; set; } = float.NaN;
    public string Tipo_de_Vivienda { get; set; } = string.Empty;
    public float Coste_USD { get; set; } = float.NaN;
    public float Año_de_Construcción { get; set; } = float.NaN;
}