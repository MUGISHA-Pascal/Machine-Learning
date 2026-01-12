const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");
const app = express();
app.use(express.static("public"));
app.use(bodyParser.json());

app.post("/prediction", async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000/predict", {
      features: req.body.features,
    });
    res.json(response.data);
    console.log(response.data.prediction);
  } catch (error) {
    console.log(`the error is ${error}`);
  }
});

app.listen(3000, () => {
  console.log("the app is running on http://localhost:3000");
});
