const { exec } = require('child_process');

// Define the command to execute
const command = 'sudo pip install faiss-cpu';

// Execute the command
exec(command, (error, stdout, stderr) => {
  if (error) {
    console.error(`Error executing command: ${error.message}`);
    return;
  }
  if (stderr) {
    console.error(`stderr: ${stderr}`);
    return;
  }
  console.log(`stdout: ${stdout}`);
});
