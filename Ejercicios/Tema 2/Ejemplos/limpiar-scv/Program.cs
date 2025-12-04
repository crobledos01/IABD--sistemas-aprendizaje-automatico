using System;
using System.Diagnostics;
using System.IO;

class ProgramMenu
{
    static void Main()
    {
        Console.WriteLine("Seleccione una opci�n:");
        Console.WriteLine("1. Ejecutar programa original");
        Console.WriteLine("2. Ejecutar ejemplo");
        
        string? input = Console.ReadLine();
        
        if (input == "1")
        {
            try
            {
                ProgramOriginal.EjecutarProgramaOriginal();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al ejecutar Mio.cs: {ex.Message}");
            }
        }
        else if (input == "2")
        {
            try
            {
                Ejemplo.EjecutarEjemplo();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al ejecutar Ejemplo.cs: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("Opci�n no v�lida.");
        }
    }
}
