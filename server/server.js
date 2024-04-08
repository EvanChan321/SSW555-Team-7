// server.mjs
import express from "express";
import bodyParser from "body-parser";
import { spawn } from "child_process";

const app = express();
const port = 3000;

app.use(express.json());

app.post("/submit-eeg", (req, res) => {
    const eegData = req.body;

    const pythonProcess = spawn("python", [
        "./server/model.py",
        JSON.stringify(eegData),
    ]);

    pythonProcess.stdout.on("data", (data) => {
        try {
            const result = JSON.parse(data);
            res.json({ probability: result.probability });
        } catch (error) {
            console.error("Error parsing JSON from python script", error);
            res.status(500).send("Error processing the EEG data.");
        }
    });

    pythonProcess.stderr.on("data", (data) => {
        console.error(`stderr: ${data}`);
        res.status(500).send(
            "An error occurred while processing the EEG data."
        );
    });
});

app.listen(port, () => console.log(`Server is running on port ${port}`));
