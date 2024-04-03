const { exec } = require('child_process');

  // Define the command to install faiss-cpu
  const installCommand = 'pip install faiss-cpu';

  // Execute the install command
  exec(installCommand, (error, stdout, stderr) => {
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
});
