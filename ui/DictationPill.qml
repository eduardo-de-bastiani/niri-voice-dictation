import QtQuick
import Quickshell
import Quickshell.Io
import Quickshell.Wayland

ShellRoot {
    id: root

    readonly property string stateFilePath: Quickshell.env("NIRI_DICTATE_UI_STATE") || ""
    readonly property string projectRoot: Quickshell.env("NIRI_DICTATE_ROOT") || ""
    readonly property string cliBin: Quickshell.env("NIRI_DICTATE_BIN") || "niri-dictate"
    readonly property string pythonBin: Quickshell.env("NIRI_DICTATE_PYTHON") || "python3"
    readonly property string audioSource: Quickshell.env("NIRI_DICTATE_SOURCE") || ""
    readonly property int bottomOffset: parseInt(Quickshell.env("NIRI_DICTATE_BOTTOM_MARGIN") || "38", 10)

    property string appState: "RECORDING"
    property string targetOutput: ""
    property list<real> micLevels: [12, 16, 22, 28, 32, 36, 32, 28, 22, 16, 12]
    property real wavePhase: 0.0
    property bool closing: false

    readonly property bool isRecording: appState === "RECORDING" && !closing
    readonly property bool isProcessing: (appState === "TRANSCRIBING" || appState === "REFINING") && !closing
    readonly property bool isError: appState === "ERROR"

    function resolveTargetScreen() {
        if (root.targetOutput !== "") {
            for (let i = 0; i < Quickshell.screens.length; i++) {
                if (Quickshell.screens[i].name === root.targetOutput)
                    return Quickshell.screens[i];
            }
        }
        return Quickshell.screens.length > 0 ? Quickshell.screens[0] : null;
    }

    function applyStateJson(rawText) {
        if (!rawText || rawText.trim() === "")
            return;
        try {
            const parsed = JSON.parse(rawText);
            if (parsed.output && typeof parsed.output === "string")
                root.targetOutput = parsed.output;
            if (parsed.state && typeof parsed.state === "string") {
                const nextState = parsed.state;
                root.appState = nextState;
                if (nextState === "COMPLETED" || nextState === "ABORTED" || nextState === "IDLE") {
                    root.beginClose(0);
                } else if (nextState === "ERROR") {
                    root.beginClose(1400);
                }
            }
        } catch (e) {
            // Ignore transient partial read; atomic os.replace will re-trigger
        }
    }

    function beginClose(delayMs) {
        if (root.closing)
            return;
        if (delayMs > 0) {
            delayedCloseTimer.interval = delayMs;
            delayedCloseTimer.restart();
        } else {
            root.closing = true;
            quitTimer.restart();
        }
    }

    Timer {
        id: delayedCloseTimer
        repeat: false
        onTriggered: {
            root.closing = true;
            quitTimer.restart();
        }
    }

    Timer {
        id: quitTimer
        interval: 190
        repeat: false
        onTriggered: Qt.quit()
    }

    // Safety watchdog: if parent state file disappears completely for >3s, exit cleanly
    Timer {
        id: orphanWatchdog
        interval: 2500
        running: true
        repeat: true
        onTriggered: {
            stateReader.reload();
            const txt = stateReader.text();
            if (!txt || txt.trim() === "")
                root.beginClose(0);
        }
    }

    FileView {
        id: stateReader
        path: root.stateFilePath
        preload: true
        watchChanges: true
        printErrors: false
        onLoaded: root.applyStateJson(stateReader.text())
        onFileChanged: {
            stateReader.reload();
            root.applyStateJson(stateReader.text());
        }
    }

    // Live PipeWire microphone amplitude streamer (runs ONLY while recording)
    Process {
        id: micMeterProcess
        running: root.isRecording && root.projectRoot !== ""
        command: [
            root.pythonBin,
            "-u",
            root.projectRoot + "/niri_dictate/mic_meter.py",
            root.audioSource
        ]

        stdout: SplitParser {
            splitMarker: "\n"
            onRead: line => {
                if (!root.isRecording || !line)
                    return;
                const parts = line.trim().split(";");
                if (parts.length < 11)
                    return;
                const next = [];
                for (let i = 0; i < 11; i++) {
                    const val = parseFloat(parts[i]);
                    next.push(isNaN(val) ? 10 : val);
                }
                root.micLevels = next;
            }
        }
    }

    // Action processes when user clicks Cancel (X) or Confirm (Check)
    Process {
        id: abortAction
        running: false
        command: [root.cliBin, "abort"]
    }

    Process {
        id: confirmAction
        running: false
        command: [root.cliBin, "toggle"]
    }

    // Smooth traveling wave animation during Transcribing / Refining
    NumberAnimation on wavePhase {
        running: root.isProcessing
        from: 0
        to: Math.PI * 2
        duration: 950
        loops: Animation.Infinite
    }

    PanelWindow {
        id: panel
        screen: root.resolveTargetScreen()
        visible: Quickshell.screens.length > 0

        anchors {
            bottom: true
        }
        margins.bottom: root.bottomOffset

        exclusionMode: ExclusionMode.Normal
        exclusiveZone: 0
        focusable: false
        aboveWindows: true
        color: "transparent"

        WlrLayershell.namespace: "niri-dictate:pill"
        WlrLayershell.layer: WlrLayer.Overlay
        WlrLayershell.keyboardFocus: WlrKeyboardFocus.None

        implicitWidth: 204
        implicitHeight: 58

        Item {
            id: pillWrapper
            anchors.centerIn: parent
            width: 192
            height: 48

            opacity: root.closing ? 0.0 : 1.0
            scale: root.closing ? 0.90 : 1.0

            Component.onCompleted: {
                opacity = 0.0;
                scale = 0.90;
                enterTimer.start();
            }

            Timer {
                id: enterTimer
                interval: 10
                repeat: false
                onTriggered: {
                    if (!root.closing) {
                        pillWrapper.opacity = 1.0;
                        pillWrapper.scale = 1.0;
                    }
                }
            }

            Behavior on opacity {
                NumberAnimation {
                    duration: 165
                    easing.type: Easing.OutCubic
                }
            }
            Behavior on scale {
                NumberAnimation {
                    duration: 185
                    easing.type: Easing.OutBack
                }
            }

            Rectangle {
                id: pillBg
                anchors.fill: parent
                radius: height / 2
                color: "#141416"
                border.width: 1
                border.color: root.isError ? "#FF5A5F" : "#2C2C32"

                Behavior on border.color {
                    ColorAnimation { duration: 150 }
                }

                // Left circular Cancel (X) button
                Rectangle {
                    id: cancelBtn
                    width: 34
                    height: 34
                    radius: 17
                    anchors.left: parent.left
                    anchors.leftMargin: 7
                    anchors.verticalCenter: parent.verticalCenter
                    color: cancelMouse.containsMouse && root.isRecording ? "#4C4C52" : "#38383D"
                    opacity: root.isRecording ? 1.0 : 0.45

                    Behavior on color { ColorAnimation { duration: 110 } }
                    Behavior on opacity { NumberAnimation { duration: 150 } }

                    // Crisp anti-aliased X icon
                    Item {
                        anchors.centerIn: parent
                        width: 14
                        height: 14

                        Rectangle {
                            anchors.centerIn: parent
                            width: 14
                            height: 2.2
                            radius: 1.1
                            color: "#EBEBEF"
                            rotation: 45
                            antialiasing: true
                        }
                        Rectangle {
                            anchors.centerIn: parent
                            width: 14
                            height: 2.2
                            radius: 1.1
                            color: "#EBEBEF"
                            rotation: -45
                            antialiasing: true
                        }
                    }

                    MouseArea {
                        id: cancelMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        enabled: !root.closing
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            abortAction.running = true;
                            root.beginClose(40);
                        }
                    }
                }

                // Center 11-bar live audio waveform / processing wave
                Row {
                    id: waveformRow
                    spacing: 4.5
                    anchors.centerIn: parent
                    height: 28

                    Repeater {
                        model: 11

                        delegate: Item {
                            required property int index
                            width: 3.2
                            height: waveformRow.height

                            readonly property real barTargetHeight: {
                                if (root.isError)
                                    return 5;
                                if (root.isProcessing) {
                                    // Smooth traveling wave while Whisper + GPT-OSS process
                                    const s = Math.sin(root.wavePhase - index * 0.52);
                                    const env = 1.0 - Math.abs(index - 5) * 0.11;
                                    return 5 + ((s + 1.0) * 0.5) * 16.0 * env;
                                }
                                const lvl = index < root.micLevels.length ? root.micLevels[index] : 10;
                                return Math.max(4.0, Math.min(26.0, 3.5 + (lvl / 100.0) * 23.0));
                            }

                            Rectangle {
                                anchors.centerIn: parent
                                width: 3.2
                                height: parent.barTargetHeight
                                radius: 1.6
                                color: root.isError ? "#FF5A5F" : (root.isProcessing ? "#D8D8E2" : "#FFFFFF")
                                antialiasing: true

                                Behavior on height {
                                    enabled: root.isRecording
                                    NumberAnimation {
                                        duration: 55
                                        easing.type: Easing.OutQuad
                                    }
                                }
                                Behavior on color {
                                    ColorAnimation { duration: 150 }
                                }
                            }
                        }
                    }
                }

                // Right circular Confirm (Check) button
                Rectangle {
                    id: confirmBtn
                    width: 34
                    height: 34
                    radius: 17
                    anchors.right: parent.right
                    anchors.rightMargin: 7
                    anchors.verticalCenter: parent.verticalCenter
                    color: confirmMouse.containsMouse && root.isRecording ? "#E6E6EC" : "#FFFFFF"
                    opacity: root.isRecording ? 1.0 : 0.65

                    Behavior on color { ColorAnimation { duration: 110 } }
                    Behavior on opacity { NumberAnimation { duration: 150 } }

                    // Crisp anti-aliased Checkmark
                    Canvas {
                        id: checkCanvas
                        anchors.fill: parent
                        antialiasing: true
                        onPaint: {
                            const ctx = getContext("2d");
                            ctx.reset();
                            ctx.strokeStyle = "#141416";
                            ctx.lineWidth = 2.4;
                            ctx.lineCap = "round";
                            ctx.lineJoin = "round";
                            ctx.beginPath();
                            ctx.moveTo(10.5, 17.5);
                            ctx.lineTo(15.0, 22.0);
                            ctx.lineTo(23.5, 12.5);
                            ctx.stroke();
                        }
                    }

                    MouseArea {
                        id: confirmMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        enabled: root.isRecording && !root.closing
                        cursorShape: root.isRecording ? Qt.PointingHandCursor : Qt.ArrowCursor
                        onClicked: {
                            confirmAction.running = true;
                        }
                    }
                }
            }
        }
    }
}
