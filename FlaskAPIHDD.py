from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.externals import joblib
import os, io, csv
import pickle
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components

app=Flask(__name__, template_folder='templates')


@app.route('/predict')
def hello_world():
    # test_json=request.get_json()
    # test = pd.read_json(test_json, orient='records')

    return render_template('hello.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result=request.form
        f=request.files['file']
        if not f:
            return "No file"

        file_data=f.read()
        print(type(file_data))
        print(file_data)

        testData=io.StringIO(unicode(file_data))
        print(testData)

        df=pd.read_csv(testData, sep=",")
        print(df)

        feature_cols=['smart_12_raw', 'smart_183_raw', 'smart_184_raw', 'smart_187_raw', 'smart_188_raw',
                      'smart_189_raw',
                      'smart_190_raw', 'smart_192_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw',
                      'smart_198_raw',
                      'smart_199_raw', 'smart_1_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw', 'smart_4_raw',
                      'smart_5_raw', 'smart_7_raw', 'smart_9_raw']
        # test=pd.read_csv("C:/Project/Dataset/checkdataset.csv", header=0)

        failureColumn=df['failure']

        loadedModel=pickle.load(open('C:/Project/HardDriveTestModel/hard_drive_model.pkl', 'rb'))
        result_pred=loadedModel.predict(df[feature_cols])
        prediction_series=list(pd.Series(result_pred))

        final_predictions=pd.DataFrame(list(zip(failureColumn, prediction_series)))
        df['Prediction']=result_pred

        hddFailure_values=df['failure'].unique()
        hddfailurevalues=[]
        hddfailure_counts=[]
        for e in hddFailure_values:
            hddfailurevalues.append(str(e))
            count=df[df['failure'] == e]['failure'].count()
            hddfailure_counts.append(count)

        p=figure(x_range=hddfailurevalues, plot_height=250, title="Hard Drive Failure Counts",
                 toolbar_location=None, tools="")

        p.vbar(x=hddfailurevalues, top=hddfailure_counts, width=0.9)

        p.xgrid.grid_line_color=None
        p.y_range.start=0
        # show(p)
        script1, div1=components(p)

        predicted_failure_values=df['Prediction'].unique()
        predicted_failurevalues=[]
        predicted_failure_counts=[]
        for e in predicted_failure_values:
            predicted_failurevalues.append(str(e))
            predicted_count=df[df['Prediction'] == e]['Prediction'].count()
            predicted_failure_counts.append(predicted_count)

        q=figure(x_range=predicted_failurevalues, plot_height=250, title="Predicted HDD Failure Counts",
                 toolbar_location=None, tools="")

        q.vbar(x=predicted_failurevalues, top=predicted_failure_counts, width=0.9)

        q.xgrid.grid_line_color=None
        q.y_range.start=0

        script2, div2=components(q)


        df.to_csv('outputResult.csv', index=False)
        responses=jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code=200

        print(list(df))

        return render_template('result.html', result=df.to_html(), the_div1=div1, the_script1=script1,  the_div2=div2, the_script2=script2)



if __name__ == '__main__':
    app.run(debug=True)
