# Sistema Inteligente de Detección de Reportes Duplicados

Repositorio para el Trabajo de Fin de Máster de Inteligencia Artificial por la UNIR (SIDRD - Sistema Inteligente de Detección de Reportes Duplicados)

# Primera vez

1. Clonar el repositorio:
   `git clone https://github.com/Alburrito/sidrd-project.git`
2. Levantar la base de datos en una terminal:
   `docker-compose up`
Esta base de datos es ahora accesible en la dirección `http://localhost:27017/`


# Uso

Para obtener un mensaje de ayuda, ejecutar, estando en `backend/`:

   ```bash
   pipenv shell
   python3 main.py -h
   ```

## Scrapear reportes

Rellenar un fichero .csv con la configuración del scraper (ver scraper_config.csv.example para un ejemplo).

Teniendo la base de datos levantada, ejecutar:
   
   ```bash
   pipenv shell
   python3 main.py -s <nombre_del_fichero>
   ```
Este fichero debe encontrarse en la carpeta `backend/scraper/config/`

Se guardarán los reportes en la base de datos y aparecerán los resultados de cada lote en un fichero .csv nuevo. en la misma carpeta

## CRUD reportes

...
