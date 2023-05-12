import time
import click
from DataCleaning.DataCleaning import datacleaning
from DataTransformed.Transformed import datatransformed

'''
Para la ejecución de del pipeline se tiene las siguientes opciones:

1. Para ejecutar todo el pipeline, se ejecuta por terminal: python .\main.py
2. Para ejecutar alguna etapa en especifico, se ejecuta por terminal, por ejemplo: 
    python .\main.py --job=datacleaning
    python .\main.py --job=transformed 
    
'''

def run_job_datacleaning():
    """
    Run job Datacleaning
    """
    try:
        datacleaning()
    except Exception as e:
        print(e)
        

def run_job_transformed():
    """
    Run job Datacleaning
    """
    try:
        datatransformed()
    except Exception as e:
        print(e)
        
        
@click.command()
@click.option(
    "--job",
    help="datacleaning, transformed",
    required=False,
)

def main(job):
    #start = time.time()
    if job is None:
        run_job_datacleaning()
        run_job_transformed()
    else:
        function_dict = {
            "datacleaning": run_job_datacleaning,
            "transformed": run_job_transformed
        }    
        job_function = function_dict.get(job)
        job_function()

if __name__ == "__main__":
    # allows to set the aws access key
    main(auto_envvar_prefix="X")  # pylint: disable=E1123,E1120