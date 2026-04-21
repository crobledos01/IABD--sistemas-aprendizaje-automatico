var estudiantes = new[] {
    new { estudia = true, duermeBien = true, entregaTareas = true, aprueba = true },
    new { estudia = true, duermeBien = false, entregaTareas = true, aprueba = true },
    new { estudia = false, duermeBien = true, entregaTareas = true, aprueba = true },
    new { estudia = false, duermeBien = false, entregaTareas = false, aprueba = false },
    new { estudia = true, duermeBien = true, entregaTareas = false, aprueba = true },
    new { estudia = false, duermeBien = false, entregaTareas = true, aprueba = false }
};

/*  De primeras, las mejores variables para empezar podrían ser "estudia" y "duermeBien", ya que en ambos casos hay un if que no
encadenaría otro interior, mientras que "entregaTareas" tiene casos verdaderos y falsos con aprobados y suspensos
Por último, me quedaría con "estudia" como la variable más importante, ya que tiene una relación más clara con el resultado
*/

var aprobados = new List<dynamic>();
var suspensos = new List<dynamic>();
var error = false;

foreach (var estudiante in estudiantes)
{
    if (estudiante.estudia)
    {
        aprobados.Add(estudiante);
        if(!estudiante.aprueba)
        {
            error = true;
            System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de aprobado");
        }
    }
    else
    {
        if (estudiante.duermeBien)
        {
            aprobados.Add(estudiante);
        
            if(!estudiante.aprueba)
            {
                error = true;
                System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de aprobado");
            }
        }
        else
        {
            suspensos.Add(estudiante);
        
            if(estudiante.aprueba)
            {
                error = true;
                System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de suspenso");
            }
        }
    }
}

if (aprobados.Count + suspensos.Count == estudiantes.Length && !error)
{
    System.Console.WriteLine("Todos los estudiantes del primer grupo han sido clasificados correctamente.");
}

var aprobados2 = new List<dynamic>();
var suspensos2 = new List<dynamic>();
var error2 = false;

foreach (var estudiante in estudiantes)
{
    if (estudiante.duermeBien)
    {
        aprobados2.Add(estudiante);
        
        if(!estudiante.aprueba)
        {
            error2 = true;
            System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de aprobado");
        }
    }
    else
    {
        if (estudiante.estudia)
        {
            aprobados2.Add(estudiante);
        
            if(!estudiante.aprueba)
            {
                error2 = true;
                System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de aprobado");
            }
        }
        else
        {
            suspensos2.Add(estudiante);
        
            if(estudiante.aprueba)
            {
                error2 = true;
                System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de suspenso");
            }
        }
    }
}


if (aprobados2.Count() + suspensos2.Count() == estudiantes.Length && !error2)
{
    System.Console.WriteLine("Todos los estudiantes del segundo grupo han sido clasificados correctamente.");
}