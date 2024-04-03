const { exec } = require('child_process');

// Define the command to check pip version
const versionCommand = 'pip --version';

// Execute the version command
exec(versionCommand, (versionError, versionStdout, versionStderr) => {
  if (versionError) {
    console.error(`Error checking pip version: ${versionError.message}`);
    return;
  }
  if (versionStderr) {
    console.error(`stderr: ${versionStderr}`);
    return;
  }

  console.log(`Pip version: ${versionStdout}`);

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
