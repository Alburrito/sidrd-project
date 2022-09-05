# Sistema Inteligente de Detección de Reportes Duplicados

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![Repo Size](https://img.shields.io/github/repo-size/Alburrito/sidrd-project)


Repositorio para el Trabajo de Fin de Máster de Inteligencia Artificial por la UNIR (SIDRD - Sistema Inteligente de Detección de Reportes Duplicados)

# Requisitos
Se requiere tener instalado al menos `python3.8`, `pipenv`, `docker`, `docker-compose`, `git` y `git-lfs`

# Instalación y uso

1. Clonar el repositorio:
   ```git clone https://github.com/Alburrito/sidrd-project.git```
2. Levantar la base de datos en una terminal:
   ```sidrd-project/$ docker-compose up```
Esta base de datos es ahora accesible en la dirección `http://localhost:27017/`

3. Instalar los paquetes necesarios, estando en `backend/`:
```bash
sidrd-project/backend/$ pipenv install
```

4. Abrir una shell
```bash
sidrd-project/backend/$ pipenv shell
```

5. Ejecutar el script de carga de datos
```bash
(sidrd-project) sidrd-project/backend/$ python3 load_data.py
```

## Ayuda

Para obtener un mensaje de ayuda, ejecutar, estando en `backend/`:

   ```bash
   (backend)sidrd-project/backend/$ python3 main.py -h
   ```

## Scrapear reportes

Rellenar un fichero .csv con la configuración del scraper (ver scraper_config.csv.example para un ejemplo).

Teniendo la base de datos levantada, ejecutar:
   
   ```bash
   sidrd-project/backend/$ pipenv shell
   (backend)sidrd-project/backend/$ python3 main.py -s <nombre_del_fichero>
   ```
Este fichero debe encontrarse en la carpeta `backend/scraper/config/`

Se guardarán los reportes en la base de datos y aparecerán los resultados de cada lote en un fichero .csv nuevo. en la misma carpeta

## Crear reporte nuevo usando SIDRD

Teniendo la base de datos levantada, ejecutar:

Ejecutar el siguiente comando:
```bash
sidrd-project/backend/$ pipenv shell
(backend)sidrd-project/backend/$ python3 main.py -c
```

El propio script te guiará por el proceso de creación de un nuevo reporte, pidiendo componente, resumen y descripción del reporte.

Proporcionará el SIDRD por defecto (utiliza stemming y resumen+componente) pero ofrece también el último SIDRD entrenado.

Al final del proceso se puede elegir si guardar el reporte en la base de datos o no, además de marcar un duplicado o no.

## Reentrenar el SIDRD
Rellenar un fichero .json con la configuración del reentreno (ver sidrd_retrain_config.json.sample para un ejemplo).

Teniendo la base de datos levantada, ejecutar:

Ejecutar el siguiente comando:
```bash
sidrd-project/backend/$ pipenv shell
(backend)sidrd-project/backend/$ python3 main.py -r <fichero_de_configuración>
```

El propio script te guiará por el proceso de reentrenamiento del SIDRD, pidiendo el fichero de configuración, ofreciendo un modo verboso para seguir el reentreno.

## Tests

Para ejecutar los tests:

```bash
sidrd-project/backend/$ pipenv shell
(sidrd-project) sidrd-project/backend/$ pytest -v
```