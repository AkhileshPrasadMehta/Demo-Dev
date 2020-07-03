import os
import joblib
import io
import flask
import pandas as pd


model_path = os.environ["MODEL_PATH"]


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
        ScoringService.get_model() is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print(request.form.to_dict())
    
    with open(os.path.join(model_path, "model.pkl"),'rb') as f:
        model = joblib.load(f)
        
    data = request.form.to_dict()
    

    for i in data:
        if i not in ['isholiday','Type']:
            if i not in ['Store','Dept','Size','Month','Type_A','Type_B','Type_C']:
                data[i] = float(data[i])
            else:
                data[i] = int(data[i])
        data[i] = [data[i]]
    

    test = pd.DataFrame(data)
    test.isholiday = test.isholiday == 'True'
    pred = model.predict(test)

    print(pred)
    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")
