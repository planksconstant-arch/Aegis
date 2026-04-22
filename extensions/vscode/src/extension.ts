import * as vscode from 'vscode';
import axios from 'axios';

const BRIDGE_URL = 'http://127.0.0.1:8765';

export function activate(context: vscode.ExtensionContext) {
    console.log('Aegis Agent is now active.');

    let startCmd = vscode.commands.registerCommand('aegis.startSession', async () => {
        vscode.window.showInformationMessage('Aegis: Connected to Local Agent Daemon.');
    });

    // Automatically send diagnostics to Aegis when a file is saved
    let saveHook = vscode.workspace.onDidSaveTextDocument(async (document) => {
        const diagnostics = vscode.languages.getDiagnostics(document.uri);
        const errorStrings = diagnostics.map(d => d.message);

        if (errorStrings.length > 0) {
            try {
                // Sent observing telemetry to local bridge
                const response = await axios.post(`${BRIDGE_URL}/tick`, {
                    task: "Fix linter errors",
                    open_files: [document.fileName],
                    diagnostics: errorStrings,
                    user_present: true
                });

                const decision = response.data.decision;
                
                if (decision && decision.action) {
                    const actionName = decision.action.payload.strategy_name;
                    vscode.window.showInformationMessage(`Aegis generated fix: ${actionName}`, 'Accept', 'Reject')
                    .then(selection => {
                        if (selection === 'Accept') {
                            vscode.commands.executeCommand('aegis.acceptSuggestion');
                        } else if (selection === 'Reject') {
                            vscode.commands.executeCommand('aegis.rejectSuggestion');
                        }
                    });
                }
            } catch (error) {
                console.error("Aegis connection error:", error);
            }
        }
    });

    let acceptCmd = vscode.commands.registerCommand('aegis.acceptSuggestion', async () => {
        await axios.post(`${BRIDGE_URL}/feedback`, { accept: true, user_present: true });
        vscode.window.showInformationMessage('Aegis learned from your acceptance.');
    });

    let rejectCmd = vscode.commands.registerCommand('aegis.rejectSuggestion', async () => {
        await axios.post(`${BRIDGE_URL}/feedback`, { accept: false, user_present: true });
        vscode.window.showInformationMessage('Aegis learned from your rejection.');
    });

    context.subscriptions.push(startCmd, saveHook, acceptCmd, rejectCmd);
}

export function deactivate() {}
