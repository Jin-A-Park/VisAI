from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn, asyncio, threading, webbrowser, websockets, json
from pydantic import BaseModel
import requests
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torch

ip_address = ''
web_client: [WebSocket] = None
start_training_event = threading.Event()
server = None

resume_flag = True
loss_buffer: list = []
accuracy_buffer: list = []
weight_buffer: list = []
activation_buffer: list = []
grad_norm_buffer: list = []
log_buffer: list = []

app = FastAPI()
app.mount('/static', StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="./")

#----------http----------
@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})
  
@app.post('/send-data')
async def get_data(data: list[dict]):
  global web_client, resume_flag
  global loss_buffer, accuracy_buffer, weight_buffer, activation_buffer, grad_norm_buffer, log_buffer

  key_to_buffer = {
    'loss_group': loss_buffer,
    'class_group': accuracy_buffer,
    'weight_group': weight_buffer,
    'activation_group': activation_buffer,
    'grad_norm_group': grad_norm_buffer,
    'log_group': log_buffer,
  }

  if not data or not isinstance(data[0], dict): return {"error": "Invalid data format"}
  for key, buffer in key_to_buffer.items():
    if key in data[0]:
      buffer.append(data)
      if web_client is not None and resume_flag:
        if buffer:
          final_buffer = buffer.pop(0)
          await asyncio.sleep(0.1)
          await web_client.send_text(json.dumps(final_buffer))
      break

  return {"status": "Data processed"}

#----------websocket & connection----------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
  global web_client, resume_flag
  web_client = websocket
  await websocket.accept()
  try:
    while True:
      data = await websocket.receive_text()
      print(f"Server received: {data}")
      if data == 'Training started': start_training_event.set()
      elif data == "Training stopped": resume_flag = False
      elif data == "Training resumed": resume_flag = True
  except WebSocketDisconnect: print("Web client disconnected")

def open_browser(host):
  global ip_address
  ip_address = host
  webbrowser.open("http://" + host + ":8000")
    
def start_server(host: str):
  threading.Timer(1, open_browser, args=(host,)).start()
  config = uvicorn.Config(app, host=host, port=8000)
  global server
  server = uvicorn.Server(config)
  server.run()
    
def connect(host: str):
  server_thread = threading.Thread(target=start_server, args=(host,))
  server_thread.start()   
  start_training_event.wait() #학습 시작 신호 -> 나머지 코드 실행

def disconnect():
  if server: server.should_exit = True
        
#----------API----------
def get_loss(n_epochs, i, train_loss, valid_loss):
  global ip_address
  
  loss_data = []
  url = f"http://{ip_address}:8000/send-data"
  loss_data.append({
    'loss_group': 'training_loss',
    'total_epoch': n_epochs,
    'epoch': i,
    'value': train_loss
  })
  loss_data.append({
    'loss_group': 'valid_loss',
    'total_epoch': n_epochs,
    'epoch': i,
    'value': valid_loss
  })
  response = requests.post(url, json=loss_data)
  
def get_acc(n_epochs, epoch, y_hat_train, target_train, y_hat_valid, target_valid):
  global ip_address
  
  acc_data = []
  url = f"http://{ip_address}:8000/send-data"

  y_hat_np = np.array(y_hat_train)
  target_np = np.array(target_train)
  if y_hat_np.shape[0] != target_np.shape[0]: raise ValueError(f"Shape mismatch: y_hat has {y_hat_np.shape[0]} samples, target has {target_np.shape[0]} samples")
  
  # Multiclass vs. multilabel for training
  if len(y_hat_np.shape) > 1 and y_hat_np.shape[1] > 1: y_hat_binary = np.argmax(y_hat_np, axis=1)
  else: y_hat_binary = (y_hat_np >= 0.5).astype(int)
  
  conf_matrix = confusion_matrix(target_np, y_hat_binary)
  class_wise_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
  target_classes = np.unique(target_np)
  
  # Validation accuracy
  y_hat_valid_np = np.array(y_hat_valid)
  target_valid_np = np.array(target_valid)
  if y_hat_valid_np.shape[0] != target_valid_np.shape[0]:
    raise ValueError(f"Shape mismatch: y_hat has {y_hat_valid_np.shape[0]} samples, target has {target_valid_np.shape[0]} samples")
  
  # Multiclass vs. multilabel for validation
  if len(y_hat_valid_np.shape) > 1 and y_hat_valid_np.shape[1] > 1: y_hat_valid_binary = np.argmax(y_hat_valid_np, axis=1)
  else:   y_hat_valid_binary = (y_hat_valid_np >= 0.5).astype(int)
  
  conf_valid_matrix = confusion_matrix(target_valid_np, y_hat_valid_binary)
  class_wise_valid_accuracy = conf_valid_matrix.diagonal() / conf_valid_matrix.sum(axis=1)
  target_valid_classes = np.unique(target_valid_np)

  # Append train accuracies
  for i, acc in enumerate(class_wise_accuracy):
    acc_data.append({
      'class_group': int(target_classes[i]),
      'total_epoch': n_epochs,
      'epoch': epoch,
      'group': 'train_accuracy',
      'value': float(acc) if not np.isnan(acc) else 0.0
    })
  
  # Append validation accuracies
  for i, acc in enumerate(class_wise_valid_accuracy):
    acc_data.append({
      'class_group': int(target_valid_classes[i]),
      'total_epoch': n_epochs,
      'epoch': epoch,
      'group': 'valid_accuracy',
      'value': float(acc) if not np.isnan(acc) else 0.0
    })

  response = requests.post(url, json=acc_data)

def get_weight(n_epochs, epoch, weights):
  global ip_address
  
  url = f"http://{ip_address}:8000/send-data"

  target_rows, target_cols = 15, 15
  if isinstance(weights, torch.Tensor): weights = weights.detach().numpy()
  rows, cols = weights.shape
  if rows < target_rows or cols < target_cols:
    labeled_weight_data = [
      {
        'weight_group': 'weight',
        'total_epoch': n_epochs,
        'epoch': epoch,
        'x': i,
        'y': j,
        'value': float(weights[i, j]),
      }
      for i in range(rows)
      for j in range(cols)
    ]
  else:
    block_rows = rows // target_rows
    block_cols = cols // target_cols
    most_active_weights = np.zeros((target_rows, target_cols))
    for i in range(target_rows):
      for j in range(target_cols):
        block = weights[i * block_rows:(i + 1) * block_rows, j * block_cols:(j + 1) * block_cols]
        most_active_weights[i, j] = block[np.unravel_index(np.abs(block).argmax(), block.shape)]
      labeled_weight_data = [
        {
          'weight_group': 'weight',
          'total_epoch': n_epochs,
          'epoch': epoch,
          'x': i,
          'y': j,
          'value': float(most_active_weights[i, j]),
        }
        for i in range(target_rows)
        for j in range(target_cols)
      ]
  response = requests.post(url, json=labeled_weight_data)
    
def get_act(n_epochs, epoch, activation_output):
  global ip_address
  
  labeled_activation_data = []
  url = f"http://{ip_address}:8000/send-data"
  
  transform = activation_output.detach().numpy()
  for i in range(transform.shape[0]):
    labeled_activation_data.append({
      'activation_group': 'activation',
      'total_epoch': n_epochs,
      'epoch': epoch,
      'activation_value': float(transform[i])
    })
  response = requests.post(url, json=labeled_activation_data)

def get_gradnorm(n_epochs, epoch, grad_norm):
  global ip_address
  
  gradient_data = []
  url = f"http://{ip_address}:8000/send-data"
  
  gradient_data.append({
    'grad_norm_group': 'grad_norm',
    'total_epoch': n_epochs,
    'epoch': epoch,
    'gradient_norm_value': grad_norm
  })
  response = requests.post(url, json=gradient_data)
  
def get_log(message):
  global ip_address
  
  log_message = []
  url = f"http://{ip_address}:8000/send-data"
  
  log_message.append({
    'log_group': 'log',
    'log_message': message
  })
  response = requests.post(url, json=log_message)
