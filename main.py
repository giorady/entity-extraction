import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from google.cloud import bigquery
import functions_framework 

@functions_framework.cloud_event
def entity_extraction(cloud_event):
    data = cloud_event.data
    bucket = data["bucket"]
    name = data["name"]

    vertexai.init(project="datalabs-int-bigdata", location="asia-southeast1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409")

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 32,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    }

    uri = "gs://"+bucket+"/"+name

    document = Part.from_uri(
    mime_type="application/pdf",
    uri=uri
    )

    prompt = """Extract Reseller Legal Name, Reseller Name, Reseller Address, and the duration term of the products ordered in Year format if its longer or equal to 12 months, from the file and give the output in a format like this without any "json" string indicator in the beginning of your response. Just follow the format:
    "reseller_legal_name": "XXXX",
    "reseller_name": "XXXX",
    "reseller_address": "XXXX",
    "term": "XX YEAR",
    """

    responses = model.generate_content(
        [document,prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )

    result = responses.text
    result_json = json.loads(result)
    json_rows=[result_json]

    table_id = "datalabs-int-bigdata.gio_dev.entity_extraction_sample"
    client = bigquery.Client()
    job = client.load_table_from_json(json_rows, table_id)
    job.result()
    # TEST
