"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
const axios_1 = require("axios");
const BRIDGE_URL = 'http://127.0.0.1:8765';
function activate(context) {
    console.log('Aegis Agent is now active.');
    let startCmd = vscode.commands.registerCommand('aegis.startSession', async () => {
        vscode.window.showInformationMessage('Aegis: Connected to Local Agent Daemon.');
    });
    let generateFixCmd = vscode.commands.registerCommand('aegis.generateFix', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage('Aegis: Open a file to generate a fix.');
            return;
        }
        const document = editor.document;
        const diagnostics = vscode.languages.getDiagnostics(document.uri);
        const errorStrings = diagnostics.map(d => d.message);
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Aegis: Generating fix via LLM & RL...",
            cancellable: false
        }, async () => {
            try {
                const response = await axios_1.default.post(`${BRIDGE_URL}/generate-fix`, {
                    task: "Fix current issues and optimize",
                    open_files: [document.fileName],
                    diagnostics: errorStrings,
                    user_present: true,
                    file_content: document.getText()
                });
                const diff = response.data.diff;
                if (diff) {
                    const edit = new vscode.WorkspaceEdit();
                    const fullRange = new vscode.Range(document.positionAt(0), document.positionAt(document.getText().length));
                    edit.replace(document.uri, fullRange, diff);
                    await vscode.workspace.applyEdit(edit);
                    vscode.window.showInformationMessage(`Aegis applied fix! (RL Confidence Score: ${response.data.score.toFixed(3)})`);
                }
                else {
                    vscode.window.showInformationMessage('Aegis could not generate a safe fix.');
                }
            }
            catch (error) {
                vscode.window.showErrorMessage("Aegis bridge error. Is the local daemon running?");
            }
        });
    });
    // Automatically send diagnostics to Aegis when a file is saved
    let saveHook = vscode.workspace.onDidSaveTextDocument(async (document) => {
        const diagnostics = vscode.languages.getDiagnostics(document.uri);
        const errorStrings = diagnostics.map(d => d.message);
        if (errorStrings.length > 0) {
            try {
                // Sent observing telemetry to local bridge
                const response = await axios_1.default.post(`${BRIDGE_URL}/tick`, {
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
                        }
                        else if (selection === 'Reject') {
                            vscode.commands.executeCommand('aegis.rejectSuggestion');
                        }
                    });
                }
            }
            catch (error) {
                console.error("Aegis connection error:", error);
            }
        }
    });
    let acceptCmd = vscode.commands.registerCommand('aegis.acceptSuggestion', async () => {
        await axios_1.default.post(`${BRIDGE_URL}/feedback`, { accept: true, user_present: true });
        vscode.window.showInformationMessage('Aegis learned from your acceptance.');
    });
    let rejectCmd = vscode.commands.registerCommand('aegis.rejectSuggestion', async () => {
        await axios_1.default.post(`${BRIDGE_URL}/feedback`, { accept: false, user_present: true });
        vscode.window.showInformationMessage('Aegis learned from your rejection.');
    });
    context.subscriptions.push(startCmd, generateFixCmd, saveHook, acceptCmd, rejectCmd);
}
function deactivate() { }
//# sourceMappingURL=extension.js.map