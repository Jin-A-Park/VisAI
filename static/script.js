const ws = new WebSocket("ws://localhost:8000/ws");

let loss_list = [];
let grad_norm_list = [];

ws.addEventListener('open', () => {
  const message = "WebSocket connection established.";
  ws.send(message);
  console.log("Client:", message);
});

ws.addEventListener("error", (event) => {
  console.error("Client: WebSocket error - ", event.message);
});

ws.addEventListener("close", () => {
  console.log("Client: WebSocket connection closed.");
});

function sendMessage(message){
  if (ws.readyState === WebSocket.OPEN) { ws.send(message); }//console.log("Client sent:", message); 
  else { console.error("WebSocket is not open. Current state:", ws.readyState); }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('start_button').addEventListener('click', () => {
    const message = "Training started";
    if (ws) { ws.send(message); }
    document.getElementById("start_button").disabled = true;
    document.getElementById("stop_button").disabled = false;
  });

  document.getElementById('stop_button').addEventListener('click', () => {
    const message = "Training stopped";
    if (ws) { ws.send(message); }
    document.getElementById("stop_button").disabled = true;
    document.getElementById("resume_button").disabled = false;
  });

  document.getElementById('resume_button').addEventListener('click', () => {
    const message = "Training resumed";
    if (ws) { ws.send(message); }
    document.getElementById("resume_button").disabled = true;
    document.getElementById("stop_button").disabled = false;
  });
});

ws.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);

  if ('loss_group' in message[0]) {
    loss_list = loss_list.concat(message);
    training_loss_update(loss_list);
  }

  if ('class_group' in message[0]){ accuracy_by_class(message); }

  if('weight_group' in message[0]){ extract_weight(message) }

  if('activation_group' in message[0]){ extract_activation(message) }

  if('grad_norm_group' in message[0]){
    grad_norm_list = grad_norm_list.concat(message)
    extract_gradient_norm(grad_norm_list)
  }

  if('log_group' in message[0]){ console.log("log from AI model ", message[0].log_message) }
});

function training_loss_update(data) {
  const vlSpec = {
    width: 500,
    height: 150,
    data: { values: data },
    mark: "line",
    encoding: {
      x: { field: "epoch", type: "quantitative", title: "Epochs" },
      y: { field: "value", type: "quantitative", title: "Training Loss" },
      color: {field: 'loss_group', type: "nominal", title: 'Loss Type'}
    }
  };
  vegaEmbed('#line_training_loss', vlSpec);
}

function accuracy_by_class(data) {
  const vlSpec = {
    width: 500,
    height: 150,
    data: { values: data },
    mark: "bar",
    encoding: {
      x: { field: "class_group", type: "nominal", title: "Class" },
      y: { field: "value", type: "quantitative", title: "Accuracy" },
      xOffset: { field: "group" },
      color: { field: "class_group", type: "nominal" },
      opacity: {
        field: "group",
        type: "nominal",
        scale: {
          domain: ["train_accuracy", "valid_accuracy"],
          range: [0.7, 0.4]
        },
        title: "Accuracy Type"
      }
    }
  };
  vegaEmbed('#bar_accuracy_by_class', vlSpec)
}

function extract_weight(data){
  const vlSpec = {
    width: 500,
    height: 150,
    data: { values: data },
    mark: {
      type: 'rect',
      tooltip: true
    },
    encoding: {
      x: {
        field: 'x',
        type: 'ordinal',
        title: 'Neurons',
      },
      y: {
        field: 'y',
        type: 'ordinal',
        title: 'Weights',
      },
      color: {
        field: 'value',
        type: 'quantitative',
        title: 'Weight Value',
        scale: { scheme: 'redblue', domainMid: 0 }
      },
      opacity: {
        value: 0.8
      }
    }
  };
  vegaEmbed('#heatmap_extract_weight', vlSpec);
}

function extract_activation(data){
  const vlSpec = {
    width: 500,
    height: 150,
    data: { values: data },
    mark: "bar",
    encoding: {
      x: {
        bin: {
          maxbins: 100,
          extent: [-0.1, 0.1],
          binned: false
        },
        field: "activation_value",
        type: 'quantitative',
        title: 'Activation Value',
        scale: {
          domain: [-0.1, 0.1],
          nice: false
        }
      },
      y: {
        aggregate: "count",
        title: 'Count',
        axis: { grid: true }
      },
    }
  };
  vegaEmbed('#histogram_extract_activation', vlSpec);
}

function extract_gradient_norm(data){
  const vlSpec = {
    width: 500,
    height: 150,
    data: { values: data },
    mark: "line",
    encoding: {
      x: { field: "epoch", type: "quantitative", title: "Epochs" },
      y: { field: "gradient_norm_value", type: "quantitative", title: "Gradient Norm" }
    }
  };
  vegaEmbed('#line_extract_gradient_norm', vlSpec);
}