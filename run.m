% - inFileName:  without ".wav" extension
% - outFileName: with ".wav" extension
% - speed:       the normal speed-up speed
function run(inFileName, outFileName, speed)

  close all;

  showPlot = false;
  visibility = 'off';
  if showPlot
    visibility = 'on';
  end
  %h = figure('visible', visibility); hold on;

  % Load music (as row vector)
  [original, Fs] = wavread([inFileName '.wav']);
  original = original(:, 1)';
  audioLength = length(original) / Fs * 1000; % audio length in ms.

  % Run harma syllable segmentation
  [syllableStarts, syllableEnds] = detectSyllables(original, Fs);
  
  % Round timestamp to align with video frames (30 fps)
  %[syllableStarts, syllableEnds] = alignToFrame(syllableStarts, syllableEnds);

  % Calculate instant syllable speed.
  density = calcDensity(original, Fs, syllableStarts, syllableEnds);

  % Filter the syllable density by median.
  densityMedian = calcDensityMedian(density);
  
  % Find valleys for segmentation.
  segPoints = calcSegments(densityMedian);

  % Merge segments.
  segments = mergeSegments(segPoints);

  % calculate speed ratio
  [starts, ends, ratios] = calcRatios(segments, densityMedian, speed,...
                                      audioLength);

  % output ratio to file
%  of = fopen(outFileName, 'w');
%  for m = 1:length(starts)
%    fprintf(of, '%d %d %.2f\n', starts(m), ends(m), ratios(m));
%  end
%  fclose(of);

  % plot
  if showPlot
    figure();
    subplot(3, 1, 1);
    plot(original);
    set(gca, 'XTick', []);
    %xlim([1, 7 * Fs]);
    ylabel('Amptitude');
    title('Audio Signal');

    %plot(density, 'g');
    subplot(3, 1, 2);
    plot(densityMedian, 'LineWidth', 3);
    set(gca, 'XTick', []);
    %xlim([1, 7000]);
    %ylim([0, 3]);
    ylabel('Number of votes');
    title('Syllable density');
    %plot(segPoints, 2 * ones(1, length(segPoints)), 'bo');
    %plot(syllableStarts * 1000, 4 * ones(1, length(syllableStarts)), 'go');
    %plot(syllableEnds * 1000, 3 * ones(1, length(syllableEnds)), 'rx');
    tmp_x = [];
    tmp_y = [];
    for m = 1:length(starts)
      tmp_x = [tmp_x, starts(m), ends(m)];
      tmp_y = [tmp_y, ratios(m) * [1, 1]];
    end
    subplot(3, 1, 3);
    plot(tmp_x, tmp_y, 'LineWidth', 3);
    %xlim([1, 7000]);
    xlabel('Time (ms)');
    ylabel('Ratio');
    title('Speed-up ratio');
  end

%  % Calculate start, end, and syllable density
%  segThresTime = 0.2; % in seconds
%  densities = [];
%  starts = [];
%  ends = [];
%  pauses = [];
%  segmentStart = syllableStarts(1);
%  syllableCount = 1;
%  for m = 2:length(syllableStarts)
%    if syllableStarts(m) - syllableEnds(m - 1) > segThresTime
%      density = syllableCount / (syllableEnds(m - 1) - segmentStart);
%      densities = [densities density];
%      starts = [starts uint32(segmentStart * Fs)];
%      ends = [ends uint32(syllableEnds(m - 1) * Fs)];
%      pauses = [pauses uint32((syllableStarts(m) - syllableEnds(m - 1)) * Fs)];
%
%      segmentStart = syllableStarts(m);
%      syllableCount = 1;
%    else
%      syllableCount = syllableCount + 1;
%    end
%  end
%  density = syllableCount / (syllableEnds(end) - segmentStart);
%  densities = [densities density];
%  starts = [starts uint32(segmentStart * Fs)];
%  ends = [ends uint32(syllableEnds(end) * Fs)];
%
%  % Calculate desired ratio of speeding up each segment and pause
%  speedRatio = 1.2; % desired speed ratio
%  speed = mean(densities) * speedRatio; % desired peak per second
%  pauseTime = 0.15; % desired length of pause (in second)
%  pauseLen = time2samples(pauseTime, Fs); % desired length of pause (#samples)
%  ratios = zeros(1, length(densities) + length(pauses));
%  ratios(1:2:end) = speed ./ densities;
%  ratios(2:2:end) = pauses ./ pauseLen;
%
%  % Round speedup ratio to multiple of 1/60
%  fps = round(1 ./ ratios .* 60);
%  ratios = 60 ./ fps;
%
%  % Merge segments and pauses to output
%  out = zeros(1, length(original));
%  writePos = 1; % position of next writing to output
%  for m = 1:length(starts)
%    % speed up segments
%    segment = original(starts(m):ends(m));
%    fastSegment = psola(segment', Fs, ratios(2 * m - 1));
%    wavwrite(fastSegment, Fs, ['pf1/' int2str(2 * m - 2) '-out.wav']);
%    [writePos, out] = appendOutput(fastSegment, out, writePos);
%    % speed up pauses
%    if m < length(starts)
%      pause = original(ends(m):starts(m + 1));
%      fastPause = psola(pause', Fs, ratios(2 * m));
%      wavwrite(fastPause, Fs, ['pf1/' int2str(2 * m - 1) '-out.wav']);
%      [writePos, out] = appendOutput(fastPause, out, writePos);
%    end
%  end
%  out = out(1:writePos - 1); % trim output

%  wavwrite(out, Fs, [fileName '-out.wav']);
%  h = figure('visible', visibility); hold on;
%  plot(out);
%  saveas(h, [fileName '-out.jpg']);

%  % Output segments and speed-ratio for video acceleration.
%  startsMs = 1000 * starts / Fs; % start time in millisecond
%  endsMs = 1000 * ends / Fs; % start time in millisecond
%
%  of = fopen(['pf1/' fileName '-segments'], 'w');
%  for m = 1:length(startsMs)
%    fprintf(of, '%d %d %d\n', startsMs(m), endsMs(m), fps(2 * m - 1));
%    if m < length(startsMs)
%      fprintf(of, '%d %d %d\n', endsMs(m), startsMs(m + 1), fps(2 * m));
%    end
%  end
%  fclose(of);

  % PSOLA usage
  %y_slow = psola(original', Fs, 0.8);
  %y_fast = psola(original', Fs, 2);
  %wavwrite(y_slow, Fs, 'slow.wav');
  %wavwrite(y_fast, Fs, 'fast_2.wav');

end

% Return number of samples (rounded down to integer) that holds
% the desired time (in seconds).
function samples = time2samples(time, Fs)
  samples = floor(time * Fs);
end

% Append src to out. Keep track of the position of writing.
function [writeEnd, out] = appendOutput(src, out, writePos)
  writeEnd = writePos + length(src);
  out(writePos:writeEnd - 1) = src;
end

% Detect syllable using Harma method, return the start/end time of syllables.
% AUDIO is a row vector.
function [syllableStarts, syllableEnds] = detectSyllables(audio, Fs)
  [syllables, Fs, S, F, T, P] =...
      harmaSyllableSeg(audio', Fs, kaiser(128), 64, 128, 20);
  syllableStarts = zeros(1, length(syllables));
  syllableEnds = zeros(1, length(syllables));
  for m = 1:length(syllables)
    syllableStarts(m) = min(syllables(m).times);
    syllableEnds(m) = max(syllables(m).times);
  end
  syllableStarts = sort(syllableStarts);
  syllableEnds = sort(syllableEnds);
end

% Align syllable start/end timestamp to multiple of video frame. (30 fps)
function [starts, ends] = alignToFrame(starts, ends)
  for m = 1:length(starts)
    starts(m) = alignTimestamp(starts(m));
    ends(m) = alignTimestamp(ends(m));
  end
end

% Align a timestamp to multiple of video frame (30 fps)
function timestamp = alignTimestamp(timestamp)
  fps = 30;
  timestamp = round(timestamp * fps) / fps;
end
 
% Calculate instant syllable speed.
% Each syllable casts 1 vote to every nearby time frame (unit: 1ms)
% within voteWindow of start and end time of the syllable.
function density = calcDensity(audio, Fs, syllableStarts, syllableEnds)
  voteWindow = 0.3; % in second
  density = zeros(1, floor(length(audio) / Fs * 1000)); % length in ms
  for m = 1:length(syllableStarts)
    % calculate the range to cast vote. Clamp to (1, end) respetively.
    voteStart = floor((syllableStarts(m) - voteWindow) * 1000);
    voteEnd = floor((syllableEnds(m) + voteWindow) * 1000);
    if voteStart < 1
      voteStart = 1;
    end
    if voteEnd > length(density)
      voteEnd = length(density);
    end
    % cast vote
    for n = voteStart:voteEnd
      density(n) = density(n) + 1;
    end
  end
end

function densityMedian = calcDensityMedian(density)
  windowSize = 151;
  densityMedian = medfilt1(density, windowSize);
end

function segPoints = calcSegments(density)
  segPoints = [1];
  inValley = false;
  valleyStart = 1;
  for m = 2:length(density)
    if density(m - 1) < density(m)
      if inValley % valley ends
        segPoints = [segPoints, valleyStart, m];
      end
      inValley = false;
    elseif density(m - 1) > density (m)
      valleyStart = m;
      inValley = true;
    end
  end
  % Make sure 'segPoints' has the end point of 'density'
  if segPoints(end) ~= length(density)
    segPoints = [segPoints length(density)];
  end
end

function segments = mergeSegments(segPoints)
  minSegmentLength = 400; % unit: ms
  segments = [1];
  segStart = segPoints(1);
  for m = 2:length(segPoints)
    if segPoints(m) - segStart > minSegmentLength
      segments = [segments segPoints(m)];
      segStart = segPoints(m);
    end
  end
end

function [starts, ends, ratios] = calcRatios(segments, density, speed,...
                                             audioLength)
  starts = [];
  ends = [];
  ratios = [];
  pauseTime = 150; % desired pause time length (unit: ms)
  avgDensity = mean(density);

  % pause count and speak time
  pauseCount = 0;
  speakTime = 0;
  for m = 2:length(segments)
    segStart = segments(m - 1);
    segEnd = segments(m) - 1; % TODO should this overlap?
    if isPause(density(segStart:segEnd))
      pauseCount = pauseCount + 1;
    else
      ratio = avgDensity / mean(density(segStart:segEnd));
      speakTime = speakTime + (segEnd - segStart) / ratio;
    end
  end
  % calculate desired ratio
  expectTime = audioLength / speed;
  desiredRatio = speakTime / (expectTime - pauseCount * pauseTime);
  % speed up
  for m = 2:length(segments)
    segStart = segments(m - 1);
    segEnd = segments(m) - 1; % TODO should this overlap?
    starts = [starts segStart];
    ends = [ends segEnd];
    % ratio
    if isPause(density(segStart:segEnd))
      ratio = (segEnd - segStart) / pauseTime;
    else
      ratio = avgDensity * desiredRatio / mean(density(segStart:segEnd));
    end
    ratios = [ratios ratio];
  end
end

function flag = isPause(segment)
  pauseThres = 0.5; % half of segment is zero
  if nnz(segment) / length(segment) < pauseThres
    flag = true;
  else
    flag = false;
  end
end
