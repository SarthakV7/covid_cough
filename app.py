import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from functions import *
import base64
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Loading model

model = load_model()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def decode_base64(data, altchars=b'+/'):
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data, altchars)


def parse_contents(contents, filename, date):
    if '.wav' not in filename:
        m1 = 'file should be in .wav format'
        print('please pass an audio file name ending with .wav')
    else:
        if 'pos' in filename:
            file_name = './pos_samples/' + filename
        else:
            file_name = './neg_samples/' + filename
        encode_string = bytes(contents.split(',')[1], 'utf-8')
        wav_file = open("temp.wav", "wb")
        decode_string = base64.b64decode(encode_string)
        wav_file.write(decode_string)
        x_img = process_data('./temp.wav')
        prob = model.predict(x_img)
        m1 = f'probability of covid-19: {prob[0][0]}'
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Audio(src=contents, controls=True),
        html.Hr(),
        html.H5("Model predictions"),
        html.H6(m1),
        ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)
