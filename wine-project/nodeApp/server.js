const axios = require("axios");
const express = require("express");
const bodyparser = require("body-parser");
const app = express();
app.use(express.static("public"));
app.use(bodyparser.json());

app.post("/prediction", async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000", {
      features: req.body.features,
    });
    res.json(response.data);
    console.log(response.data.prediction);
  } catch (error) {
    console.log(error);
  }
});
app.listen(3000, () => {
  console.log("app is runnning on http://localhost:3000");
});
