const express = require("express");
const axios = require("axios");
const bodyparser = require("body-parser");

const app = express();
app.use(bodyparser.json());
app.use(express.static("public"));

app.post("/prediction", async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000/predict", {
      features,
    });
    res.json(response.data);
    console.log(response.data.prediction);
  } catch (error) {
    console.log(error);
  }
});

app.listen(3000, () => {
  console.log("app running on http://localhost:3000");
});
