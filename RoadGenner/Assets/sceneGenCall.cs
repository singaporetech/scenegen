using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using TMPro;
using UnityEngine.XR;
using OVR;

public class sceneGenCall : MonoBehaviour
{
    public TMP_InputField inputField;
    public Button generateButton;
    public RawImage displayImage;
    public Material skyboxMaterial;
    public GameObject uiCanvas;

    void Start()
    {
        generateButton.onClick.AddListener(OnGenerateButtonClicked);
    }

    void Update()
    {
        if (OVRInput.GetDown(OVRInput.Button.One))
        {
            ToggleUI();
        }
    }

    void OnGenerateButtonClicked()
    {
        string inputText = inputField.text;
        StartCoroutine(PostRequest(" PUT SERVER PREDICTION ENDPOINT HERE", inputText));
    }

    IEnumerator PostRequest(string url, string inputText)
    {

        using (UnityWebRequest www = UnityWebRequest.Post(url, "{\"prompt\":\"" + inputText + "\"}", "application/json"))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError(www.error);
            }
            else
            {
                Debug.Log(inputText);
                string responseText = www.downloadHandler.text;
                string base64String = ExtractBase64String(responseText);
                Texture2D texture = Base64ToTexture(base64String);

                if (texture != null)
                {
                    skyboxMaterial.SetTexture("_MainTex", texture);
                    RenderSettings.skybox = skyboxMaterial;

                    // Hide the UI after setting the skybox
                    uiCanvas.SetActive(false);
                }
            }
        }
    }
    string ExtractBase64String(string response)
    {
        var jsonObj = JsonUtility.FromJson<ServerResponse>(response);
        string base64String = jsonObj.image_uri;

        // Remove the prefix if it exists
        if (base64String.Contains("base64,"))
        {
            base64String = base64String.Substring(base64String.IndexOf("base64,") + 7);
        }

        return base64String;
    }
    Texture2D Base64ToTexture(string base64String)
    {
        byte[] imageBytes = System.Convert.FromBase64String(base64String);
        Texture2D texture = new Texture2D(2048, 1024, TextureFormat.RGB24, false);
        texture.LoadImage(imageBytes);
        texture.Apply(false, true); // Disable mipmaps
        return texture;
    }

    void ToggleUI()
    {
        uiCanvas.SetActive(!uiCanvas.activeSelf);
    }

    [System.Serializable]
    public class ServerResponse
    {
        public string image_uri;
    }
}
