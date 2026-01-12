const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const app = express();
app.use(bodyParser.json());
app.use(express.static("public"));

app.post("/prediction", async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000/predict", {
      features: req.body.features,
    });
    res.json(response.data);
    console.log(req.body.features);
    console.log(response.data.prediction);
  } catch (error) {
    console.log(`the error : ${error}`);
  }
});
app.listen(3000, () => {
  console.log("the app is running on http://localhost:3000");
});
